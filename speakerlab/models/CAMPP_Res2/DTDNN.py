# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from speakerlab.models.campplus.layers import DenseLayer, StatsPool, TDNNLayer, CAMDenseTDNNBlock, TransitLayer, BasicResBlock, get_nonlinear, BasicBlockERes2Net_diff_AFF


class FCM(nn.Module):
    def __init__(self,
                block=BasicBlockERes2Net_diff_AFF, # 把一个残差块的类作为默认参数传进来
                num_blocks=[3, 4], ## 与ERes2Net前两个block相匹配 # 用于设置每个layer中使用几个
                m_channels=32,
                feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels

        # [B,1,F,T] -> [B,m_channels,F,T]
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        # m_channels:输出通道数
        # [B,m_channels,F,T] -> [B,m_channels*expansion,F/2,T]
        self.layer1 = self._make_layer_ERes2Block(block, m_channels, num_blocks[0], stride=2) 

        # [B,m_channels*expansion,F/2,T] -> [B,m_channels*2*expansion,F/4,T]
        self.layer2 = self._make_layer_ERes2Block(block, m_channels * 2, num_blocks[1], stride=2)

        # [B,m_channels*2*expansion,F/4,T] -> [B,m_channels,F/8,T]
        self.conv2 = nn.Conv2d(self.in_planes, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels =  m_channels * (feat_dim // 8)

        def _make_layer_ERes2Block(self, block, planes, num_blocks, stride):
        """
        构建stage，包含多个block
        planes：每个 block 的目标通道
        num_blocks：block 数量
        stride：该stage中第一层的步幅，用于下采样

        return:该stage的各blockSequential
        """
        # 生成 strides 列表，第一个 block 可能执行下采样（stride=2），之后的 block 都 stride=1
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride)) # 每个block在最后的1x1卷积中返回的通道数是self.expansion*planes
            self.in_planes = planes * block.expansion # 更新下一个block的输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        # [B,m_channels,F/8,T] -> [B,m_channels*(F/8),1,T]
        out = out.reshape(shape[0], shape[1]*shape[2], shape[3]) # 展平频率维度和通道维度
        return out

class CAMPPlus(nn.Module):
    def __init__(self,
                 feat_dim=80,
                 embedding_size=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels # self.out_channels =  m_channels * (feat_dim // 8)

        self.xvector = nn.Sequential(
            OrderedDict([

                ('tdnn',
                 TDNNLayer(channels,
                           init_channels,
                           5,
                           stride=2, # 2.2 Additionally, we adopted an input TDNN layer with 1/2 subsampling rate before the D-TDNN backbone to accelerate computation.”
                           dilation=1,
                           padding=-1,
                           config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size,
                dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers,
                                   in_channels=channels,
                                   out_channels=growth_rate,
                                   bn_channels=bn_size * growth_rate,
                                   kernel_size=kernel_size,
                                   dilation=dilation,
                                   config_str=config_str,
                                   memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate # 过完一个CAMDenseTDNNBlock后通道数
            self.xvector.add_module( # 过渡层
                'transit%d' % (i + 1),
                TransitLayer(channels,
                             channels // 2,
                             bias=False,
                             config_str=config_str))
            channels //= 2

        self.xvector.add_module(
            'out_nonlinear', get_nonlinear(config_str, channels))

        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module(
            'dense',
            DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))

        for m in self.modules(): # Kaiming 正态分布初始化权重
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        return x
