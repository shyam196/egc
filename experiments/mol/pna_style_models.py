"""The models here are supposed to correspond to those used by the PNA paper"""
import abc

import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from experiments.layers import EfficientGraphConv, Mpnn
from experiments.utils import mlp


class HIVNet(nn.Module, abc.ABC):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        readout="mean",
        activation=nn.ReLU,
        norm=nn.BatchNorm1d,
    ):
        super().__init__()
        self.embedding = AtomEncoder(hidden_dim)
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


class GcnHIVNet(HIVNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        return GCNConv(hidden_dim, hidden_dim)


class GatHIVNet(HIVNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        readout="mean",
        activation=nn.ReLU,
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
            in_feat_drop=in_feat_drop,
            residual=residual,
            readout=readout,
            activation=activation,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        # holdover from the Benchmarking GNNs paper where they found this useful
        if layer_idx == self.num_graph_layers - 1:
            heads = 1
        else:
            heads = self.heads

        ctor = GATConv if self.gat_version == 1 else GATv2Conv
        return ctor(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=self.gat_dropout,
        )


class GinHIVNet(HIVNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        return GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True)


class EgcHIVNet(HIVNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        readout="mean",
        activation=nn.ReLU,
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
            aggrs=self.aggrs,
        )


class MpnnHIVNet(HIVNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        readout="mean",
        activation=nn.ReLU,
        aggr="add",
    ):
        self.mpnn_aggr = aggr
        super().__init__(
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            in_feat_drop=in_feat_drop,
            residual=residual,
            readout=readout,
            activation=activation,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        return Mpnn(self.mpnn_aggr, hidden_dim, hidden_dim)


class SageHIVNet(HIVNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        # SAGE aggr is usually mean, which is the default here
        return SAGEConv(hidden_dim, hidden_dim)
