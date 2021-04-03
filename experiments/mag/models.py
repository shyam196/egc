"""Based on OGB repo"""
import torch
import torch.nn.functional as F

from experiments.optimized_layers import EGConv


IN_FEATURES = 128
OUT_ROUNDED = 352
OUT_TRUE = 349


# Lightly modified from the OGB repo


class EGC(torch.nn.Module):
    def __init__(
        self, hidden_channels, num_layers, dropout, num_heads, num_bases, aggrs
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            EGConv(
                IN_FEATURES,
                hidden_channels,
                aggrs=aggrs,
                num_heads=num_heads,
                num_bases=num_bases,
                cached=True,
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                EGConv(
                    hidden_channels,
                    hidden_channels,
                    aggrs=aggrs,
                    num_heads=num_heads,
                    num_bases=num_bases,
                    cached=True,
                )
            )
        self.convs.append(
            EGConv(
                hidden_channels,
                OUT_ROUNDED,
                aggrs=aggrs,
                num_heads=num_heads,
                num_bases=num_bases,
                cached=True,
            )
        )

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # The output is actually slightly bigger than needed, since we have this
        # head/base stuff; hence we will truncate it.
        x = self.convs[-1](x, adj_t)[:, :OUT_TRUE]
        return x.log_softmax(dim=-1)
