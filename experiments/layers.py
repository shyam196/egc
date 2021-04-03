import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul


class EfficientGraphConv(nn.Module):
    """The EGC Layer described in the paper"""
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        num_bases,
        softmax_weights,
        add_self_loops=True,
        bias=True,
        aggrs=None,
        cache=False,
        sigmoid_weights=False,
        hardtanh_weights=False,
        **kwargs,
    ):
        super().__init__()
        assert aggrs is not None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.softmax_weights = softmax_weights
        self.add_self_loops = add_self_loops
        self.sigmoid_weights = sigmoid_weights
        self.hardtanh_weights = hardtanh_weights

        if softmax_weights:
            assert not sigmoid_weights and not hardtanh_weights
        elif sigmoid_weights:
            assert not softmax_weights and not hardtanh_weights
        elif hardtanh_weights:
            assert not softmax_weights and not sigmoid_weights

        assert self.out_channels % self.num_heads == 0

        self.comb_weights = nn.Linear(
            in_channels,
            num_heads * num_bases * len(aggrs),
        )

        # You can fuse this into one weight matrix.
        # I didn't just to make the implementation easier for other readers
        self.bases_weight = nn.ParameterList(
            [
                nn.Parameter(
                    torch.Tensor(
                        self.in_channels, (self.out_channels // self.num_heads)
                    )
                )
                for _ in range(self.num_bases)
            ]
        )

        # NOTE: in the interest of implementing things quickly, and avoiding bugs
        # aggregator fusion is _not_ used for the numerical experiments.
        # Instead, we use "_AggLayer"s that are message passing layers
        # that do nothing except apply the aggregator after message passing.
        self.aggs = nn.ModuleList()
        for a in aggrs:
            self.aggs.append(_AggLayer(a, add_self_loops=add_self_loops, cache=cache))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.comb_weights.reset_parameters()
        for w in self.bases_weight:
            glorot(w)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index):
        # NOTE: There is almost certainly a better way to implement this!
        # This implementation is the closest match to what is described in the paper.
        # In the interests of reproducibility we retain the original implementation.
        # One possible alternative is to reshape appropriately, and use a batched
        # matrix multiplication (per node) to do the weighting per-head.

        # define L = O // H
        bases = []
        for w in self.bases_weight:
            bases.append(torch.matmul(x, w))  # N x L
        bases = torch.stack(bases, dim=1)  # N x B x L
        bases = bases.view(bases.shape[0], -1)  # N x BL

        aggregated = []
        for a in self.aggs:
            y = a(x=bases, edge_index=edge_index)  # N x BL
            y = y.view(y.shape[0], self.num_bases, -1)  # N x B x L
            aggregated.append(y)

        y = torch.stack(aggregated, dim=2)  # N x B x A x L
        weightings = self.comb_weights(x)  # N x HBA

        if self.softmax_weights:
            weightings = weightings.view(
                -1, self.num_heads, self.num_bases * len(self.aggs)
            )  # N x H x BA
            # NOTE: applying the softmax over all basis/aggregators!
            weightings = weightings.softmax(dim=-1)
            weightings = weightings.view(
                -1, self.num_heads, self.num_bases, len(self.aggs), 1
            )  # N x H x B x A x 1
        else:
            if self.sigmoid_weights:
                weightings = F.sigmoid(weightings)
            elif self.hardtanh_weights:
                weightings = F.hardtanh(weightings)

            weightings = weightings.view(
                -1, self.num_heads, self.num_bases, len(self.aggs), 1
            )  # N x H x B x A x 1

        y = y.unsqueeze(1)  # N x 1 x B x A x L

        z = weightings * y  # N x H x B x A x L
        z = torch.sum(z, dim=(2, 3))  # N x H x L
        z = z.view(-1, self.out_channels)  # N x O

        if self.bias is not None:
            z = z + self.bias

        return z

    def extra_repr(self):
        return (
            f"(In={self.in_channels}, Out={self.out_channels}, H={self.num_heads}, "
            + f"B={self.num_bases}, SL={self.add_self_loops}, SM={self.softmax_weights}, "
            + f"Bias={self.bias is not None})"
        )


class _AggLayer(MessagePassing):
    """Message passing layer that does nothing except apply aggregators"""
    def __init__(self, aggr, add_self_loops, cache):
        self.aggr_fun = aggr
        if aggr == "min":
            aggr = "max"
        elif aggr == "symadd":
            aggr = "add"
        elif aggr in ["var", "std"]:
            aggr = None
        super().__init__(aggr=aggr)
        self.add_self_loops = add_self_loops
        self.cache = cache
        self.cached_vals = None

    def forward(self, x, edge_index):
        edge_weight = None
        if self.aggr_fun == "symadd":
            if self.cache and self.cached_vals is not None:
                edge_index, edge_weight = self.cached_vals

            else:
                if isinstance(edge_index, torch.Tensor):
                    edge_index, edge_weight = gcn_norm(
                        edge_index=edge_index,
                        edge_weight=None,
                        num_nodes=x.shape[0],
                        add_self_loops=self.add_self_loops,
                    )
                elif isinstance(edge_index, SparseTensor):
                    edge_index = gcn_norm(
                        edge_index=edge_index,
                        edge_weight=None,
                        num_nodes=x.shape[0],
                        add_self_loops=self.add_self_loops,
                    )

                if self.cache:
                    self.cached_vals = (edge_index, edge_weight)

        if self.aggr_fun == "min":
            return -self.propagate(x=-x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            return self.propagate(x=x, edge_index=edge_index, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        else:
            return x_j

    def aggregate(self, inputs, index, ptr, dim_size):
        if self.aggr_fun in ["var", "std"]:
            mean = scatter(
                inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="mean"
            )
            mean_sq = scatter(
                inputs * inputs,
                index,
                dim=self.node_dim,
                dim_size=dim_size,
                reduce="mean",
            )
            out = mean_sq - mean * mean
            if self.aggr_fun == "std":
                out = torch.sqrt(torch.relu(out) + 1e-5)
            return out

        else:
            return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

    def message_and_aggregate(self, adj_t, x):
        if self.aggr_fun in ["var", "std"]:
            # TODO
            raise NotImplementedError
        return matmul(adj_t, x, reduce=self.aggr)

    def extra_repr(self):
        return self.aggr_fun


class Mpnn(MessagePassing):
    """Baseline MPNN"""
    def __init__(self, aggr, in_dim, out_dim, towers=4):
        super().__init__(aggr=aggr)
        assert out_dim % towers == 0 and in_dim % towers == 0
        self.message_layer = nn.ModuleList(
            [nn.Linear(2 * in_dim // towers, out_dim // towers) for _ in range(towers)]
        )
        self.update_layer = nn.ModuleList(
            [nn.Linear(2 * out_dim // towers, out_dim // towers) for _ in range(towers)]
        )
        self.lin = nn.Linear(out_dim, out_dim)

        self.towers = towers
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x, edge_index):
        return self.lin(self.propagate(x=x, edge_index=edge_index, x_init=x))

    def message(self, x_j, x_i):
        x_i = x_i.reshape(-1, self.towers, self.in_dim // self.towers)
        x_j = x_j.reshape(-1, self.towers, self.in_dim // self.towers)
        h = torch.cat([x_i, x_j], dim=-1)
        out = []
        for i, l in enumerate(self.message_layer):
            out.append(l(h[:, i]))
        return torch.cat(out, dim=-1)

    def update(self, inputs, x_init):
        inputs = inputs.reshape(-1, self.towers, self.in_dim // self.towers)
        x_init = x_init.reshape(-1, self.towers, self.in_dim // self.towers)
        h = torch.cat([inputs, x_init], dim=-1)
        out = []
        for i, l in enumerate(self.update_layer):
            out.append(l(h[:, i]))
        return torch.cat(out, dim=-1)


if __name__ == "__main__":
    layer = EfficientGraphConv(128, 128, 4, 8, False)
    print(layer)
