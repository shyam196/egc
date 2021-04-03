import abc

import torch.nn as nn
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    GATv2Conv,
)

from experiments.layers import EfficientGraphConv
from experiments.utils import mlp

IN_FEATURES = 28


class ZincNet(nn.Module, abc.ABC):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        readout="mean",
        norm=nn.BatchNorm1d,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.embedding = nn.Embedding(IN_FEATURES, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_drop)

        self.graph_layers = nn.ModuleList()
        self.num_graph_layers = num_graph_layers

        for i in range(num_graph_layers):
            self.graph_layers.append(
                nn.ModuleList(
                    [
                        self.make_graph_layer(hidden_dim, i),
                        norm(hidden_dim),
                        activation(),
                    ]
                )
            )

        if readout == "mean":
            self.pool = global_mean_pool
        elif readout == "sum":
            self.pool = global_add_pool
        elif readout == "max":
            self.pool = global_max_pool
        else:
            raise ValueError

        self.mlp = mlp(
            [hidden_dim, hidden_dim // 2, hidden_dim // 4, 1], act=activation
        )
        self.residual = residual

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.embedding(x.squeeze())
        x = self.in_feat_dropout(x)

        for gcn, bn, act in self.graph_layers:
            identity = x
            x = gcn(x=x, edge_index=edge_index)
            x = bn(x)
            x = act(x)
            if self.residual:
                x = x + identity

        x = self.pool(x, batch.batch)
        return self.mlp(x)

    @abc.abstractmethod
    def make_graph_layer(self, hidden_dim, layer_idx):
        raise NotImplementedError


class Gatv2ZincNet(ZincNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        # Carry-over from the Benchmarking GNNs paper
        heads = 8 if layer_idx != self.num_graph_layers - 1 else 1
        return GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
        )


class EgcZincNet(ZincNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        readout="mean",
        activation=nn.ReLU,
        heads=8,
        bases=4,
        softmax=False,
        sigmoid=False,
        hardtanh=False,
        aggrs=None,
    ):
        assert aggrs is not None
        self.heads = heads
        self.bases = bases
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.hardtanh = hardtanh
        self.aggrs = aggrs

        super().__init__(
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            in_feat_drop=in_feat_drop,
            residual=residual,
            readout=readout,
            activation=activation,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        return EfficientGraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=self.heads,
            num_bases=self.bases,
            softmax_weights=self.softmax,
            hardtanh_weights=self.hardtanh,
            sigmoid_weights=self.sigmoid,
            aggrs=self.aggrs,
        )
