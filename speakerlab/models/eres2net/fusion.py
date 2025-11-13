# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn


class AFF(nn.Module):

    def __init__(self, channels=64, r=4):
        """
        channels:输入特征的通道数
        r:压缩比例
        """
        super(AFF, self).__init__()
        inter_channels = int(channels // r) # 计算压缩后的通道数量

        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0), # 第一个1x1卷积，输入通道数为两个要融合的通道数（channels * 2），压缩到inter_channels
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0), # 第二个1x1卷积，通道数恢复为channels，表示权重
            nn.BatchNorm2d(channels),
        )

    def forward(self, x, ds_y):
        """
        融合x 和 ds_y
        """
        xa = torch.cat((x, ds_y), dim=1)

        # 计算注意力权重
        x_att = self.local_att(xa)

        x_att = 1.0 + torch.tanh(x_att)

        # 1+x_att 应用在x上，1-x_att应用在ds_y上，相加
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0-x_att)

        return xo

