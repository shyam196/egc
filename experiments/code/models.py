import abc

import torch.nn as nn
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    PNAConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from experiments.code.utils import (
    MAX_DEPTH,
    NUM_NODEATTRIBUTES_1,
    NUM_NODEATTRIBUTES_2,
    NUM_NODETYPES,
    SEQ_LEN,
    VOCAB_SIZE,
)
from experiments.layers import EfficientGraphConv, Mpnn


class ASTNodeEncoder(nn.Module):
    """Taken from OGB Repo"""

    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return (
            self.type_encoder(x[:, 0])
            + self.attribute_encoder(x[:, 1])
            + self.depth_encoder(depth)
        )


class CodeNet(nn.Module, abc.ABC):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        readout="mean",
        activation=nn.ReLU,
        use_old_code_dataset=False,
    ):
        super().__init__()
        self.embedding = ASTNodeEncoder(
            hidden_dim,
            num_nodeattributes=NUM_NODEATTRIBUTES_1
            if use_old_code_dataset
            else NUM_NODEATTRIBUTES_2,
            num_nodetypes=NUM_NODETYPES,
            max_depth=MAX_DEPTH,
        )
        self.in_feat_dropout = nn.Dropout(in_feat_drop)

        self.graph_layers = nn.ModuleList()
        self.num_graph_layers = num_graph_layers

        for i in range(num_graph_layers):
            self.graph_layers.append(
                nn.ModuleList(
                    [
                        self.make_graph_layer(hidden_dim, i),
                        nn.BatchNorm1d(hidden_dim),
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

        self.token_predictors = nn.ModuleList()
        for _ in range(seq_len):
            # +2 -- 1 for unknown tokens, 1 for end delimiter
            self.token_predictors.append(nn.Linear(hidden_dim, vocab_size + 2))

        self.residual = residual

    def forward(self, batch):
        x, edge_index, node_depth = batch.x, batch.edge_index, batch.node_depth

        x = self.embedding(
            x,
            node_depth.view(
                -1,
            ),
        )
        x = self.in_feat_dropout(x)

        for gcn, bn, act in self.graph_layers:
            identity = x
            x = gcn(x=x, edge_index=edge_index)
            x = bn(x)
            x = act(x)
            if self.residual:
                x = x + identity

        x = self.pool(x, batch.batch)
        preds = []
        for token_pred in self.token_predictors:
            preds.append(token_pred(x))
        return preds

    @abc.abstractmethod
    def make_graph_layer(self, hidden_dim, layer_idx):
        raise NotImplementedError


class GcnCodeNet(CodeNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        return GCNConv(hidden_dim, hidden_dim)


class GatCodeNet(CodeNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        readout="mean",
        activation=nn.ReLU,
        heads=8,
        gat_dropout=0.0,
        gat_version=1,
        use_old_code_dataset=False,
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
            vocab_size=vocab_size,
            seq_len=seq_len,
            use_old_code_dataset=use_old_code_dataset,
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


class GinCodeNet(CodeNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        return GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True)


class EgcCodeNet(CodeNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        readout="mean",
        activation=nn.ReLU,
        heads=8,
        bases=8,
        softmax=False,
        aggrs=None,
        use_old_code_dataset=False,
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
            vocab_size=vocab_size,
            seq_len=seq_len,
            use_old_code_dataset=use_old_code_dataset,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        return EfficientGraphConv(
            hidden_dim,
            hidden_dim,
            num_heads=self.heads,
            num_bases=self.bases,
            aggrs=self.aggrs,
            softmax_weights=self.softmax,
        )


class MpnnCodeNet(CodeNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        readout="mean",
        activation=nn.ReLU,
        aggr="add",
        use_old_code_dataset=False,
    ):
        self.mpnn_aggr = aggr
        super().__init__(
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            in_feat_drop=in_feat_drop,
            residual=residual,
            readout=readout,
            activation=activation,
            vocab_size=vocab_size,
            seq_len=seq_len,
            use_old_code_dataset=use_old_code_dataset,
        )

    def make_graph_layer(self, hidden_dim, layer_idx):
        return Mpnn(self.mpnn_aggr, hidden_dim, hidden_dim)


class PnaCodeNet(CodeNet):
    def __init__(
        self,
        hidden_dim,
        num_graph_layers,
        in_feat_drop,
        residual,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        readout="mean",
        activation=nn.ReLU,
        deg=None,
        use_old_code_dataset=False,
    ):
        assert deg is not None
        self.deg = deg
        super().__init__(
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            in_feat_drop=in_feat_drop,
            residual=residual,
            readout=readout,
            activation=activation,
            vocab_size=vocab_size,
            seq_len=seq_len,
            use_old_code_dataset=use_old_code_dataset,
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


class SageCodeNet(CodeNet):
    def make_graph_layer(self, hidden_dim, layer_idx):
        return SAGEConv(hidden_dim, hidden_dim)
