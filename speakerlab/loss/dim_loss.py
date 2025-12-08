import numpy as np
import torch
import torch.nn as nn

import math
import torch
import torch.nn.functional as F

"""
DIM 互信息损失模块（中文注释版）。

此文件实现了基于 infoNCE 的互信息估计（Noise-Contrastive Estimation），用于
拉近局部特征与对应全局特征的距离，同时抑制非对应样本之间的相似度。

主要组件：
- `MI1x1ConvNet`: 基于 1x1 卷积的投影网络，用于将局部/全局特征投影到互信息空间。
- `infonce_loss`: 实现多类（multi-global）infoNCE 损失的计算。
- `LocalDIM`: 将 local/global 特征映射后直接返回 infoNCE 损失。

注意：张量的形状说明在各函数的注释中均有说明，务必保证传入形状与注释一致。
"""

# from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation
# from cortex_DIM.nn_modules.misc import Permute

class Permute(torch.nn.Module):
    """Module for permuting axes.

    """
    def __init__(self, *perm):
        """

        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.

        Args:
            input: Input tensor.

        Returns:
            torch.Tensor: permuted tensor.

        """
        return input.permute(*self.perm)

class MI1x1ConvNet(nn.Module):
    """基于 1x1 卷积的互信息投影网络（用于特征图 / 卷积输出）

    说明：
    - 该模块接收形状 `(N, C, H, W)` 的张量，并对通道维做投影，输出维度为 `n_units`。
    - 使用 1x1 卷积等价于对每个空间位置独立地应用相同的通道变换（对空间不做混合），这在处理卷积特征图时非常常用。
    - 结构为：Conv(1x1) -> BN2d -> ReLU -> Conv(1x1)，并与一个线性 shortcut 相加，然后做 LayerNorm（通过 Permute 实现对通道维进行归一化）。

    设计细节：
    - `linear_shortcut` 尝试初始化为近似恒等（如果 `n_units >= n_input`），以便训练初期保留原始通道信息。
    - 使用 `Permute` 将张量维度临时调整为 `(N, H, W, C)`，应用 `LayerNorm(n_units)`，再把维度换回 `(N, C, H, W)`。
    """

    def __init__(self, n_input, n_units):
        """构造函数

        参数:
            n_input (int): 输入通道数 C。
            n_units (int): 输出通道数（mi_units）。
        """

        super().__init__()

        # 非线性主分支：1x1 卷积 -> BN -> ReLU -> 1x1 卷积
        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # 在通道维上应用 LayerNorm：先把通道移动到最后，再做 LayerNorm，最后再还原维度
        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        # shortcut 分支：1x1 卷积映射
        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # 尝试把 shortcut 初始化为近似恒等（当输出通道 >= 输入通道时）
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """前向计算

        参数:
            x (torch.Tensor): 输入张量，形状 `(N, C, H, W)`。

        返回:
            torch.Tensor: 输出张量，形状 `(N, n_units, H, W)`。
        """

        # 把主分支与 shortcut 相加，然后做通道方向的 LayerNorm
        h = self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))
        return h


def infonce_loss(l, m):
    """
    计算 infoNCE（Noise-Contrastive Estimation）损失（多类情况）。

    形状说明（调用约定）：
    - l: 局部特征张量，形状为 `(N, D, T)`，其中 N=batch 大小，D=通道数/单元数，T=局部位置数（n_locals）。
    - m: 多个全局特征，形状为 `(N, D, M)`，其中 M=每个样本对应的全局向量数量（n_multis），通常为 1 或多类表示。

    算法要点：
    - 对每个局部向量与其对应的全局向量计算正样本相似度（inner product），
      并把同一 batch 中其他样本的全局向量作为负样本（outer products）。
    - 为避免样本自身作为负样本，使用掩码将对角线（self 对应项）置为很小的值（在 softmax 前通过减去较大常数实现）。
    - 最终在类别维度上做 log_softmax，取正样本（拼接后第一个位置）的对数概率并求负均值作为损失。

    Returns:
        torch.Tensor: 标量损失。
    """

    # N: batch size, units: 特征维度 D, n_locals: 局部位置数 T
    N, units, n_locals = l.size()
    # m 的形状 (N, D, n_multis)
    _, _ , n_multis = m.size()

    # 先调整为 (N, T, D) 方便矩阵乘法（局部位置作为第一维的样本序列）
    l_p = l.permute(0, 2, 1)   # (N, n_locals, D)
    m_p = m.permute(0, 2, 1)   # (N, n_multis, D)

    # 展平为二维用于批量 outer-product 操作
    l_n = l_p.reshape(-1, units)   # (N * n_locals, D)
    m_n = m_p.reshape(-1, units)   # (N * n_multis, D)

    # 正样本相似度：对每个局部向量与其对应 batch 的全局向量做内积
    # u_p 形状解释：torch.matmul(l_p, m) -> (N, n_locals, n_multis)，unsqueeze(2) -> (N, n_locals, 1, n_multis)
    u_p = torch.matmul(l_p, m).unsqueeze(2)

    # 负样本相似度：使用矩阵乘法得到 (N*n_multis, N*n_locals) 的相似矩阵，然后重塑为 (N, N, n_locals, n_multis)
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)  # -> (N, N, n_locals, n_multis) -> permute -> (N, N, n_locals, n_multis)

    # 构造掩码以屏蔽掉正样本对应的 self 对应项（对角线），掩码形状为 (N, N, 1, 1)
    mask = torch.eye(N)[:, :, None, None].to(l.device)
    n_mask = 1 - mask

    # 把 self 对应项通过减一个较大常数来屏蔽（softmax 前），避免其影响负样本分布
    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples

    # 重新排列为按局部位置展开的负样本集合，形状 -> (N, n_locals, N * n_locals, n_multis)
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

    # 在类别维（dim=2）上拼接正样本与负样本 logits，然后计算 log_softmax
    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)

    # 正样本在拼接后位于类别维的第一个位置，取其对数概率并求负均值作为最终损失
    loss = -pred_log[:, :, 0].mean()

    return loss


def compute_dim_loss(l_enc, m_enc):
    
    loss = infonce_loss(l_enc, m_enc)

    return loss


class LocalDIM(nn.Module):
    def __init__(self, local_channels, global_channels, mi_units=512):
        super().__init__()
        self.local_net = MI1x1ConvNet(local_channels, mi_units)
        self.global_net = MI1x1ConvNet(global_channels, mi_units)

    def forward(self, local_feat, global_feat):
        """
        local_feat:  (B, C_local, T)
        global_feat: (B, C_global)
        """

        B, C_local, T = local_feat.shape
        B2, C_global = global_feat.shape
        assert B == B2

        # ------ reshape 成卷积可接受的形式 ------
        # local -> (B, C_local, T, 1)
        local_feat = local_feat.unsqueeze(-1)

        # global -> (B, C_global, 1, 1)
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)

        # ------ 1x1 conv map ------
        L = self.local_net(local_feat)      # (B, D, T, 1)
        G = self.global_net(global_feat)    # (B, D, 1, 1)

        # ------ reshape 为 infoNCE 需要的格式 ------
        L = L.squeeze(-1)  # -> (B, D, T)
        G = G.squeeze(-1).squeeze(-1)  # -> (B, D)

        # 扩展 global 特征到 infoNCE 形式 (B, D, 1)
        G = G.unsqueeze(-1)

        # ------ 计算 DIM 损失 ------
        loss = compute_dim_loss(L, G)

        return loss
    
class ArcMarginLoss(nn.Module):
    """
    Implement of additive angular margin loss.
    """
    def __init__(self,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False):
        super(ArcMarginLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def forward(self, cosine, label):
        # cosine : [batch, numclasses].
        # label : [batch, ].
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = torch.zeros(cosine.size()).type_as(cosine)
        one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, label)
        return loss
    

class FusionLoss(nn.Module):
    """
    融合 AAMSoftmax + DIM 的总损失
    用于 x-vector / ECAPA / CAM++ 等任意网络
    -------------------------------------------------
    使用方式：
        fusion_loss = FusionLoss(dim_module, arc_margin, lambda_dim=0.1)
        loss = fusion_loss(global_embed, local_feat, label)
    """
    def __init__(self, local_channels, global_channels, scale=32.0, margin=0.2, easy_margin=False, mi_units=512, lambda_dim=0.1):
        super().__init__()
        self.dim_module = LocalDIM(local_channels=local_channels, global_channels=global_channels, mi_units=mi_units)      # LocalDIM 实例
        self.arc_margin = ArcMarginLoss(scale=scale, margin=margin, easy_margin=easy_margin)       # ArcMarginLoss 实例
        self.lambda_dim = lambda_dim       # 权重

    def forward(self, global_embed, local_feat, label):
        """
        参数：
            global_embed: (B, D) 全局向量（x-vector or ECAPA embedding）
            local_feat:   (B, C, T) 局部特征（TDNN/CNN 中间层输出）
            label:        (B,) 说话人标签
        -------------------------------------------------
        返回：
            loss_total: AAM + λ * DIM
            loss_dict:  用于监控的多项损失
        """

        # -------- AAM loss --------
        loss_aam = self.arc_margin(global_embed, label)

        # -------- DIM loss --------
        # LocalDIM 输入：(local_feat, global_embed)
        loss_dim = self.dim_module(local_feat, global_embed)

        # -------- 总损失 --------
        loss_total = loss_aam + self.lambda_dim * loss_dim

        return loss_total, {
            "loss_total": loss_total.item(),
            "loss_aam": loss_aam.item(),
            "loss_dim": loss_dim.item()
        }