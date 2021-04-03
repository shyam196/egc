"""Mostly adapted from the R-GCN example on the OGB repo"""
import copy

import torch
import torch.nn.functional as F
from torch.nn import Parameter, ModuleDict, ModuleList, Linear, ParameterDict
from torch_geometric.nn.inits import glorot


NUM_NODES_DICT = {
    "author": 1134649,
    "field_of_study": 59965,
    "institution": 8740,
    "paper": 736389,
}
X_TYPES = ["paper"]
NODE_TYPES = ["author", "field_of_study", "institution", "paper"]
EDGE_TYPES = [
    ("author", "affiliated_with", "institution"),
    ("institution", "to", "author"),
    ("author", "writes", "paper"),
    ("paper", "to", "author"),
    ("paper", "cites", "paper"),
    ("paper", "has_topic", "field_of_study"),
    ("field_of_study", "to", "paper"),
]

IN_FEATURES = 128
NUM_CLASSES = 349


class RGCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # `ModuleDict` does not allow tuples :(
        self.rel_lins = ModuleDict(
            {
                f"{key[0]}_{key[1]}_{key[2]}": Linear(
                    in_channels, out_channels, bias=False
                )
                for key in EDGE_TYPES
            }
        )

        self.root_lins = ModuleDict(
            {key: Linear(in_channels, out_channels, bias=True) for key in NODE_TYPES}
        )

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins.values():
            lin.reset_parameters()
        for lin in self.root_lins.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        out_dict = {}
        for key, x in x_dict.items():
            out_dict[key] = self.root_lins[key](x)

        for key, adj_t in adj_t_dict.items():
            key_str = f"{key[0]}_{key[1]}_{key[2]}"
            x = x_dict[key[0]]
            out = self.rel_lins[key_str](adj_t.matmul(x, reduce="mean"))
            out_dict[key[2]].add_(out)

        return out_dict


class REGConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_bases):
        super(REGConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_bases = num_bases

        self.bases_weight = Parameter(
            torch.Tensor(in_channels, (out_channels // num_heads) * num_bases)
        )

        # `ModuleDict` does not allow tuples :(
        self.rel_combs = ModuleDict(
            {
                f"{key[0]}_{key[1]}_{key[2]}": Linear(
                    in_channels, 2 * num_heads * num_bases
                )
                for key in EDGE_TYPES
            }
        )

        self.root_combs = ModuleDict(
            {key: Linear(in_channels, num_heads * num_bases) for key in NODE_TYPES}
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.bases_weight)
        for lin in self.rel_combs.values():
            lin.reset_parameters()
        for lin in self.root_combs.values():
            lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        bases = {}
        for key, x in x_dict.items():
            bases[key] = torch.matmul(x, self.bases_weight)

        root_combined = {}
        for key, x in x_dict.items():
            weightings = self.root_combs[key](x).view(
                -1, self.num_heads, self.num_bases
            )
            root_combined[key] = torch.matmul(
                weightings,
                bases[key].view(
                    -1,
                    self.num_bases,
                    self.out_channels // self.num_heads,
                ),
            )

        for key, adj_t in adj_t_dict.items():
            # 0 = source, 2 = target
            key_str = f"{key[0]}_{key[1]}_{key[2]}"
            aggregated_mean = adj_t.matmul(bases[key[0]], reduce="mean")
            aggregated_max = adj_t.matmul(bases[key[0]], reduce="max")
            aggregated = torch.stack([aggregated_mean, aggregated_max], dim=1).view(
                -1,
                self.num_bases * 2,
                self.out_channels // self.num_heads,
            )
            weightings = self.rel_combs[key_str](x_dict[key[2]]).view(
                -1, self.num_heads, self.num_bases * 2
            )
            root_combined[key[2]] += torch.matmul(weightings, aggregated)

        for key, x in root_combined.items():
            root_combined[key] = x.view(-1, self.out_channels)

        return root_combined


class REGC(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_layers,
        dropout,
        use_egc=True,
        egc_heads=8,
        egc_bases=4,
    ):
        super(self).__init__()

        node_types = list(NUM_NODES_DICT.keys())

        self.embs = ParameterDict(
            {
                key: Parameter(torch.Tensor(NUM_NODES_DICT[key], IN_FEATURES))
                for key in set(node_types).difference(set(X_TYPES))
            }
        )

        self.convs = ModuleList()

        # NOTE: only the first layer is an EGC layer
        if use_egc:
            self.convs.append(
                REGConv(IN_FEATURES, hidden_channels, egc_heads, egc_bases)
            )
        else:
            self.convs.append(RGCNConv(IN_FEATURES, hidden_channels))

        for _ in range(num_layers - 2):
            if use_egc:
                self.convs.append(
                    REGConv(hidden_channels, hidden_channels, egc_heads, egc_bases)
                )
            else:
                self.convs.append(RGCNConv(hidden_channels, hidden_channels))

        self.convs.append(RGCNConv(hidden_channels, NUM_CLASSES))

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embs.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        x_dict = copy.copy(x_dict)
        for key, emb in self.embs.items():
            x_dict[key] = emb

        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, adj_t_dict)
            for key, x in x_dict.items():
                x = F.relu(x)
                x_dict[key] = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x_dict, adj_t_dict)
