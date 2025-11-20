# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn
from speakerlab.models.eres2net.fusion import AFF


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear

def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    def __init__(self,
                bn_channels, # 输入CAM模块中的输入维度实际上是在CAMDenseTDNNLayer中经过全连接（瓶颈层）降维的维度，即 bn_channels=bn_size * growth_rate
                out_channels,# growth_rate
                kernel_size,
                stride,
                padding,
                dilation,
                bias,
                reduction=2,
                channel_groups=8): # 通道分组数 
        super(CAMLayer, self).__init__()

        # CAM中的TDNN模块 [B, bn_channels, T] -> [B, out_channels, T]
        self.linear_local = nn.Conv1d(bn_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 bias=bias)

        # 在这里加入分组卷积定义
        # [B, bn_channels, T] -> [B, out_channels, T]
        self.conv1 = nn.Conv1d(bn_channels, out_channels, kernel_size=3, padding=1, groups=channel_groups, bias=False)

        # CAM
        # 通道注意力
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

        # 掩码融合权重alpha
        self.alpha = nn.Parameter(torch.tensor(-2.5, dtype=torch.float))

    def forward(self, x):

        # CAM中的TDNN模块 [B, bn_channels, T] -> [B, out_channels, T]
        y = self.linear_local(x)

        # 对通道池化得到 T 维向量，然后加到所有通道上，得到T维向量，将均值加到每个通道上
        # [B, bn_channels, T] -> [B, 1, T] -> broadcast add -> [B, bn_channels, T]
        context0 = x.mean(dim=1, keepdim=True) + x

        # 执行分组卷积获得掩码
        # [B, bn_channels, T] -> [B, out_channels, T] -> Sigmoid
        # m0 = self.sigmoid(self.conv1(context0))
        m0 = self.conv1(context0)


        # seg_pooling 返回[B, bn_channels, T] x.mean(-1, keepdim=True) 返回 [B, bn_channels, 1] ：相加返回 [B, bn_channels, T]
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)

        # context:[B, bn_channels, T] -> [B, bn_channels // reduction, T] -> ReLU
        context = self.relu(self.linear1(context))

        # [B, bn_channels // reduction, T] -> [B, out_channels, T] 
        # m = self.sigmoid(self.linear2(context))
        m = self.linear2(context)

        # 权重限制，使用局部变量保存 sigmoid(self.alpha)，避免覆盖 Parameter
        alpha = torch.sigmoid(self.alpha)

        # 融合掩码
        m_mix = self.sigmoid(alpha * m0 + (1 - alpha) * m)

        return y*m_mix

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        
        # x:[B, bn_channels, T] -> seg:[B, bn_channels, 「T//seg_len]
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        
        shape = seg.shape
        # unsqueeze(-1) → [B, bn_channels, T//seg_len, 1] -> expand(*shape, seg_len) → [B, bn_channels, T//seg_len, seg_len] -> reshape(*shape[:-1], -1) → [B, bn_channels, 「T//seg_len * seg_len]
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)

        # 裁剪到 T，保持与输入尺寸一致 [B, bn_channels, 「T//seg_len * seg_len] -> [B, bn_channels, T]
        seg = seg[..., :x.shape[-1]]

        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self,
                 in_channels, # in_channels=in_channels + i * out_channels
                 out_channels, # 模块的输出维度实际上就是 growth_rate 
                 bn_channels, # bn_channels=bn_size * growth_rate
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient

        # 瓶颈层定义
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)

        # 第二个非线性层定义，在CAM层之前
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        
        # CAM 层定义       
        self.cam_layer = CAMLayer(bn_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
    # 瓶颈层函数 先过BN+ReLU 再过全连接
    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        # 瓶颈层函数 先过 BN+ReLU 再过 全连接，用于降维
        # [B,F,T] （BN+ReLU) -> [B,bn_channels,T] bn_channels为瓶颈层输出通道数
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        # [B,bn_channels,T] (BN+ReLU) -> CAMLayer融合后 -> [B,out_channels,T]
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels, # growth_rate
                 bn_channels, # bn_channels=bn_size * growth_rate
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels,
                                   out_channels=out_channels,
                                   bn_channels=bn_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   dilation=dilation,
                                   bias=bias,
                                   config_str=config_str,
                                   memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    expansion = 1 # 输出通道数相对于内部 planes 的放大倍数

    def __init__(self, in_planes, planes, stride=1):
        # in_planes:输入通道数，planes:输出基本通道数，stride:在Height维度的步长
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes: # stride != 1对应空间维度不一致的情况，in_planes!=self.expansion*planes对应通道维度不一致的情况
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes, # 处理通道不一致
                          kernel_size=1, # 使用1*1卷积
                          stride=(stride, 1), # 处理空间不一致
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

