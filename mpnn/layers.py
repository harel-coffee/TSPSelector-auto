import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import BatchNorm1d, Linear
from torch.nn.modules.module import Module
from torch_geometric.nn.conv import MessagePassing

class FC(Module):
    def __init__(self, in_ch, out_ch):
        super(FC, self).__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class MLP(Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(MLP, self).__init__()
        self.fc1 = FC(in_ch, hid_ch)
        self.fc2 = FC(hid_ch, out_ch)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BatchNorm(BatchNorm1d):
    r"""Applies batch normalization over a batch of node features as described
    in the `"Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
    """
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm, self).__init__(in_channels, eps, momentum, affine,
                                        track_running_stats)
    def forward(self, x):
        """"""
        return super(BatchNorm, self).forward(x)


    def __repr__(self):
        return ('{}({}, eps={}, momentum={}, affine={}, '
                'track_running_stats={})').format(self.__class__.__name__,
                                                  self.num_features, self.eps,
                                                  self.momentum, self.affine,
                                                  self.track_running_stats)


class CGConv(MessagePassing):
    r"""The crystal graph convolutional operator from the
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
        \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s  \right)

    where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
    \mathbf{e}_{i,j} ]` denotes the concatenation of central node features,
    neighboring node features and edge features.
    In addition, :math:`\sigma` and :math:`g` denote the sigmoid and softplus
    functions, respectively.

    Args:
        channels (int): Size of each input sample.
        dim (int): Edge feature dimensionality.
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, channels, dim, aggr='add', bias=True, **kwargs):
        super(CGConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = channels
        self.out_channels = channels
        self.dim = dim

        self.lin_f = Linear(2 * channels + dim, channels, bias=bias)
        self.lin_s = Linear(2 * channels + dim, channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()


    def forward(self, x, edge_index, edge_attr):
        """"""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


    def message(self, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view((edge_attr.shape[0], 1)) if edge_attr.dim() == 1 else edge_attr
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def update(self, aggr_out, x):
        return aggr_out + x

    def __repr__(self):
        return '{}({}, {}, dim={})'.format(self.__class__.__name__,
                                           self.in_channels, self.out_channels,
                                           self.dim)

