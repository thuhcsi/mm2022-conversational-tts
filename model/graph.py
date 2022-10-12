from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param
from torch_scatter import scatter
from torch_sparse import SparseTensor, masked_select_nnz, matmul

from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn.conv.rgcn_conv import RGCNConv, masked_edge_index

def masked_edge_weight(edge_weight, edge_mask):
    return edge_weight[edge_mask]

class RGCNConv(RGCNConv):

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index: Adj, edge_type: OptTensor = None, edge_weight = None):
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            tmp2 = masked_edge_weight(edge_weight, edge_type == i)

            if x_l.dtype == torch.long:
                out += self.propagate(tmp, x=weight[i, x_l], size=size, edge_weight=tmp2)
            else:
                h = self.propagate(tmp, x=x_l, size=size, edge_weight=tmp2)
                out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class RGCNConv_FG(RGCNConv):

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index: Adj, edge_type: OptTensor = None, edge_weight = None):
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), x_r.size(1), self.out_channels, device=x_r.device)

        weight = self.weight
        for i in range(self.num_relations):
            tmp = masked_edge_index(edge_index, edge_type == i)
            tmp2 = masked_edge_weight(edge_weight, edge_type == i)

            if x_l.dtype == torch.long:
                out += self.propagate(tmp, x=weight[i, x_l], size=size, edge_weight=tmp2)
            else:
                h = self.propagate(tmp, x=x_l, size=size, edge_weight=tmp2)
                out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return x_j if edge_weight is None else torch.bmm(edge_weight, x_j)
