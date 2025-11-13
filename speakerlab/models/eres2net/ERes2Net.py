# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

""" 
    Res2Net implementation is adapted from https://github.com/wenet-e2e/wespeaker.
    ERes2Net incorporates both local and global feature fusion techniques to improve the performance. 
    The local feature fusion (LFF) fuses the features within one single residual block to extract the local signal.
    The global feature fusion (GFF) takes acoustic features of different scales as input to aggregate global signal.
"""


import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import speakerlab.models.eres2net.pooling_layers as pooling_layers
from speakerlab.models.eres2net.fusion import AFF

class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class BasicBlockERes2Net(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        """
        in_planes:输入通道数
        planes:输出通道数
        stride:第一个1x1卷积的步长
        baseWidth:用于计算3x3卷积中每个分组输入（=输出）的通道数的比率
        scale:res2netBlock中3x3卷积分组数

        return:[B,self.expansion*planes,F/stride,T/stride]
        """
        super(BasicBlockERes2Net, self).__init__()
        width = int(math.floor(planes*(baseWidth/64.0))) # width计算3x3卷积中每个分组输入（=输出）的通道数
        self.conv1 = nn.Conv2d(in_planes, width*scale, kernel_size=1, stride=stride, bias=False) # 第一个1x1卷积 
        self.bn1 = nn.BatchNorm2d(width*scale)
        self.nums = scale # 分组数量

        convs=[]
        bns=[]
        for i in range(self.nums): # 每个分组使用一个3x3卷积
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)
        
        # 第二个1x1卷积
        self.conv3 = nn.Conv2d(width*scale, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        # 捷径
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes: # 如果第一个1x1卷积做了下采用/第二个1x1卷积做了通道维度扩张，对x做相应操作
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))
            
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x # 残差

        # 1x1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3分组
        spx = torch.split(out,self.width,1)
        for i in range(self.nums):
        	if i==0: 
        		sp = spx[i] # 第一组的输入是spx[0]
            else:
        		sp = sp + spx[i] # 后面组的输入是 spx[i] + sp(上一组的卷积结果)
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i==0:
        		out = sp # 第一组的输出是自己
        	else:
        		out = torch.cat((out,sp),1) # 后面组的输出是拼接后的结果

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out

class BasicBlockERes2Net_diff_AFF(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, baseWidth=32, scale=2):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        width = int(math.floor(planes*(baseWidth/64.0)))
        self.conv1 = nn.Conv2d(in_planes, width*scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        self.nums = scale

        convs=[]
        fuse_models=[]
        bns=[]
        for i in range(self.nums):
        	convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
        	bns.append(nn.BatchNorm2d(width))
             
        # 分组数-1个AFF
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(width*scale, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out,self.width,1)     
        for i in range(self.nums):
            if i==0:
                sp = spx[i] # 第一组的输入是spx[0]
            else:
                sp = self.fuse_models[i-1](sp, spx[i]) # 后面组的输入使用AFF融合spx[i] 和 sp 来作为输入
                
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i==0:
                out = sp
            else:
                out = torch.cat((out,sp),1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out

class ERes2Net(nn.Module):
    def __init__(self,
                 block=BasicBlockERes2Net,
                 block_fuse=BasicBlockERes2Net_diff_AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=32, # 每个 stage 的初始通道数
                 feat_dim=80,
                 embedding_size=192,
                 pooling_func='TSTP', # Temporal Self-attentive Two-level Pooling
                 two_emb_layer=False):
        super(ERes2Net, self).__init__()
        self.in_planes = m_channels # 更新当前block的输入通道数
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8 # 每帧的特征维度，feat_dim有8倍下采样，m_channels加大8倍
        self.two_emb_layer = two_emb_layer

        # [B,1,F,T] -> [B,32,F,T]
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        # stage 1: [B,32,F,T] -> [B,64,F,T] m_channels=32是目标通道数，但block内做了expand
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)

        # stage 2: [B,64,F,T] -> [B,128,F/2,T/2]
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)

        # stage 3: [B,128,F/2,T/2] -> [B,256,F/4,T/4]
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2)

        # stage 4:[B,256,F/4,T/4] -> [B,512,F/8,T/8]
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2)

        # 用于全局连接（各stage之间fusion)的下采样层
        # [B,64,F,T] -> [B,128,F/2,T/2]
        self.layer1_downsample = nn.Conv2d(m_channels * 2, m_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        # [B,128,F/2,T/2] -> [B,256,F/4,T/4]
        self.layer2_downsample = nn.Conv2d(m_channels * 4, m_channels * 8, kernel_size=3, padding=1, stride=2, bias=False)
        # [B,256,F/4,T/4] -> [B,512,F/8,T/8]
        self.layer3_downsample = nn.Conv2d(m_channels * 8, m_channels * 16, kernel_size=3, padding=1, stride=2, bias=False)

        # Bottom-up fusion module  用于全局连接（各stage之间fusion)的AFF
        # stage 1 and stage 2
        self.fuse_mode12 = AFF(channels=m_channels * 4)
        # stage 12 and stage 3
        self.fuse_mode123 = AFF(channels=m_channels * 8)
        # stage 123 and stage 4
        self.fuse_mode1234 = AFF(channels=m_channels * 16)

        # n_stats 表示是否拼接了均值与标准差
        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2

        # 池化层定义：self.stats_dim表示最后一个block后每帧的特征维度（频率*通道）*expansion
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        
        # 池化后的全连接，输入维度：为最后一个block后每帧的特征维度（频率*通道）*expansion*self.n_stats
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                               embedding_size)
        
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
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
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)   
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        stats = self.pool(fuse_out1234)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


if __name__ == '__main__':

    x = torch.zeros(10, 300, 80)
    model = ERes2Net(feat_dim=80, embedding_size=192, pooling_func='TSTP')
    model.eval()
    out = model(x)
    print(out.shape) # torch.Size([10, 192])

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6)) # 6.61M

