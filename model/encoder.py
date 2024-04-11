import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul
from torch_geometric.data import HeteroData

from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, GraphConv, GATConv, GATv2Conv, HGTConv, Linear


class HeteroGraphSage(torch.nn.Module):
    def __init__(self, input_dim, dim, num_layers=2, dropout=0.0, projection=False):
        super().__init__()

        self.dim = dim

        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers

        if projection:
            self.proj = nn.Linear(input_dim, dim)

        self.convs = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.skip_lins = torch.nn.ModuleList()
        for l in range(num_layers):
            if l == 0 and not projection:
                self.convs.append(SAGEConv(input_dim, dim, root_weight=True, normalize=False))
            else:
                self.convs.append(SAGEConv(dim, dim, root_weight=True, normalize=False))
            if not projection:
                self.skip_lins.append(nn.Linear(input_dim, dim))
            else:
                self.skip_lins.append(nn.Linear(dim, dim))
            self.activations.append(nn.PReLU(dim))

    def forward(self, x, edge_index, edge_weight=None):
        if hasattr(self, 'proj'):
            x = self.proj(x)

        h = x
        for l in range(len(self.convs)):
            h = self.dropout(h)
            h = self.convs[l](h, edge_index)
            h = self.activations[l](h)
            skip = self.skip_lins[l](x)
            h = h + skip

        return h

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)

