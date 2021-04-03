import abc

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GINConv, PNAConv, SAGEConv

from experiments.layers import EfficientGraphConv, Mpnn
from experiments.utils import mlp

NUM_FEATURES = 128
NUM_CLASSES = 40


class ArxivNet(nn.Module, abc.ABC):
    def __init__(self, hidden_dim, num_graph_layers, dropout, residual):
        super().__init__()
        self.num_graph_layers = num_graph_layers
        self.embed = mlp([NUM_FEATURES, hidden_dim], dropout=dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_graph_layers):
            self.convs.append(self.make_graph_layer(hidden_dim, i))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.out = nn.Linear(hidden_dim, NUM_CLASSES)
        self.dropout = dropout
        self.residual = residual

    def forward(self, x, edge_index):
        x = self.embed(x)
        for i, conv in enumerate(self.convs):
            identity = x
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual:
                x = x + identity

        x = self.out(x)
        return x.log_softmax(dim=-1)

    @abc.abstractmethod
    def make_graph_layer(self, hidden_dim, layer_idx):
        raise NotImplementedError


class GcnArxivNet(ArxivNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        return GCNConv(hidden_dim, hidden_dim)


class GatArxivNet(ArxivNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        dropout,
        residual,
        heads=8,
        gat_dropout=0.0,
        gat_version=1,
    ):
        self.heads = heads
        self.gat_dropout = gat_dropout
        assert gat_version in [1, 2]
        self.gat_version = gat_version
        super().__init__(
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
            residual=residual,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        # holdover from the Benchmarking GNNs paper where they found this useful
        if layer_idx == self.num_graph_layers - 1:
            heads = 1
        else:
            heads = self.heads

        ctor = GATConv if self.gat_version == 1 else GATv2Conv
        return ctor(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=self.gat_dropout,
        )


class GinArxivNet(ArxivNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        return GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True)


class EgcArxivNet(ArxivNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        dropout,
        residual,
        heads=8,
        bases=8,
        softmax=False,
        aggrs=None,
    ):
        assert aggrs is not None
        self.heads = heads
        self.bases = bases
        self.softmax = softmax
        self.aggrs = aggrs

        super().__init__(
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
            residual=residual,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        return EfficientGraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=self.heads,
            num_bases=self.bases,
            softmax_weights=self.softmax,
            aggrs=self.aggrs,
        )


class MpnnArxivNet(ArxivNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        dropout,
        residual,
        aggr="add",
    ):
        self.mpnn_aggr = aggr
        super().__init__(
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
            residual=residual,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        return Mpnn(self.mpnn_aggr, hidden_dim, hidden_dim)


class PnaArxivNet(ArxivNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        dropout,
        residual,
        deg=None,
    ):
        assert deg is not None
        self.deg = deg
        super().__init__(
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            dropout=dropout,
            residual=residual,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        return PNAConv(
            hidden_dim,
            hidden_dim,
            aggregators=["mean", "min", "max", "std"],
            scalers=["identity", "amplification", "attenuation"],
            deg=self.deg,
            towers=4,
            divide_input=True,
        )


class SageArxivNet(ArxivNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        # SAGE aggr is usually mean, which is the default here
        return SAGEConv(hidden_dim, hidden_dim)
