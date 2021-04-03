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

IN_FEATURES = 5
NUM_CLASSES = 10


class CifarNet(nn.Module, abc.ABC):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        residual,
        readout="mean",
        activation=nn.ReLU,
        dropout=0.0,
        norm=nn.BatchNorm1d
    ):
        super().__init__()
        self.embedding = nn.Linear(IN_FEATURES, hidden_dim)

        self.graph_layers = nn.ModuleList()
        self.num_graph_layers = num_graph_layers

        for i in range(num_graph_layers):
            self.graph_layers.append(
                nn.ModuleList(
                    [
                        nn.Dropout(dropout),
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
            [hidden_dim, hidden_dim // 2, hidden_dim // 4, NUM_CLASSES], act=activation
        )
        self.residual = residual

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.embedding(x.squeeze())

        for drop, gcn, bn, act in self.graph_layers:
            identity = x
            x = drop(x)
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


class Gatv2CifarNet(CifarNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        # As in Benchmarking GNNs paper
        heads = 8 if layer_idx != self.num_graph_layers - 1 else 1
        return GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
        )


class EgcCifarNet(CifarNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        residual,
        readout="mean",
        activation=nn.ReLU,
        dropout=0.0,
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
            residual=residual,
            readout=readout,
            activation=activation,
            dropout=dropout,
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
