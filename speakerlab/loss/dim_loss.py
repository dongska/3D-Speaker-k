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


def infonce_loss(l, m, temperature=0.07, large_neg=-1e9, debug=False):
    """
    稳定的 infoNCE 损失（支持多 global 的情况）。

    函数输入说明：
        l: (N, D, T)  本地特征（local），例如时间步的局部向量
        m: (N, D, M)  多个全局/多视角特征（multi-global），M 表示每个样本的全局向量数量

    目标：对于每个 batch 中的每个本地向量（N * T 个），其正例为对应样本下的 M 个全局向量中的匹配对，
    其余来自其他样本的全局向量视为负例。使用 softmax + 交叉熵（等价于 infoNCE）来将正例概率最大化。

    关键点：
    - 为了数值稳定与效率，先对向量做 L2 归一化，再通过矩阵乘法批量计算相似度；
    - 通过把同一 batch 的样本对角位置（self）设置为非常小的值（large_neg）来屏蔽自身作为负例；
    - 最终在 class 轴上做 log_softmax，取正类（第 0 类）的负对数概率平均作为损失。

    参数：
        temperature: 相似度缩放系数（默认为 0.07）
        large_neg: 用于屏蔽自举项的极小值
        debug: 打印调试信息

    返回值：单个标量损失
    """
    assert l.dim() == 3 and m.dim() == 3
    N, D, n_locals = l.size()
    _, D2, n_multis = m.size()
    assert D == D2

    # 先把形状从 (N, D, T) -> (N, T, D)，(N, D, M) -> (N, M, D)，便于按特征维做归一化与 matmul
    l_p = l.permute(0, 2, 1).contiguous()  # (N, T, D)
    m_p = m.permute(0, 2, 1).contiguous()  # (N, M, D)

    # 在特征维上做 L2 归一化，使内积等价于余弦相似度（数值稳定且利于训练）
    l_p = F.normalize(l_p, p=2, dim=2)
    m_p = F.normalize(m_p, p=2, dim=2)

    # 为了使用 torch.matmul，构造 (N, D, M) 形式的 m
    m_norm = m_p.permute(0, 2, 1).contiguous()  # (N, D, M)

    # 计算正例相似度：对每个 local（N, T, D）与每个同样样本下的 global（N, D, M）做矩阵乘法
    # 结果 u_p 形状为 (N, T, M)，表示每个 local 对应 M 个正例的相似度
    u_p = torch.matmul(l_p, m_norm)  # (N, T, M)
    u_p = u_p / temperature
    if debug:
        pos = u_p.detach()
        print("pos stats:", pos.mean().item(), pos.std().item(), pos.min().item(), pos.max().item())

    # 为了高效计算所有负例相似度，将 l_p 和 m_p 展平为二维矩阵再用一次 mm
    # l_n: (N * T, D) ; m_n: (N * M, D)
    l_n = l_p.reshape(-1, D)   # (N * T, D)
    m_n = m_p.reshape(-1, D)   # (N * M, D)

    # 通过矩阵乘法得到所有 global 与所有 local 的相似度：形状 (N*M, N*T)
    u_n = torch.mm(m_n, l_n.t())
    # 把结果重塑为 (N, M, N, T) ，随后转置为 (N, N, T, M)
    # 其中第一维是 global 的源样本索引，第三维是 local 的源样本索引
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1).contiguous()

    if debug:
        print("u_n raw stats:", u_n.mean().item(), u_n.std().item())

    # 屏蔽对自身样本的负例（即当 global 来自 same-sample 时不要把它作为负例）
    # mask 形状为 (N, N, 1, 1)，主对角为 1，其余为 0
    mask = torch.eye(N, device=l.device)[:, :, None, None]
    inv_mask = 1.0 - mask

    # 对角位置（self）置为 large_neg，使其在 softmax 中概率接近 0
    u_n = u_n * inv_mask + large_neg * (1.0 - inv_mask)

    # 为了拼接到 class 维上，调整 negatives 的形状：先 reshape -> (N, N*T, M)，
    # 然后扩展为 (N, T, N*T, M) 以与正类维对齐
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(1).expand(-1, n_locals, -1, -1)

    # 把正例维度扩成 (N, T, 1, M)，确保正类在拼接时位于第一个位置
    u_p = u_p.unsqueeze(2)  # (N, T, 1, M)

    # 在 class 轴（dim=2）上拼接正/负例，得到 (N, T, 1 + N*T, M)
    pred_lgt = torch.cat([u_p, u_n], dim=2)  # (N, T, 1 + N*T, M)

    # 如果每个样本有多个 global（M>1），把最后两个维度合并成一个 class 轴：
    # (N, T, num_classes) ，num_classes = (1 + N*T) * M
    if pred_lgt.size(-1) > 1:
        pred_lgt = pred_lgt.reshape(N, n_locals, pred_lgt.size(2) * pred_lgt.size(3))

    if debug:
        print("pred_lgt shape before softmax:", pred_lgt.shape, "mean/std:", pred_lgt.mean().item(), pred_lgt.std().item())

    # 在 class 轴上做 log_softmax，并取第 0 类（正类）的对数概率作为目标
    pred_log = F.log_softmax(pred_lgt, dim=2)

    # loss = - E[ log p(correct=0) ]，对所有 (N, T) 求平均
    loss = -pred_log[:, :, 0].mean()
    return loss



class LocalDIM(nn.Module):
    def __init__(self, local_channels, global_channels, mi_units=512):
        super().__init__()
        self.local_net = MI1x1ConvNet(local_channels, mi_units)
        # create global_net only when needed
        if global_channels != mi_units:
            self.global_net = MI1x1ConvNet(global_channels, mi_units)
        else:
            self.global_net = None
        self.mi_units = mi_units

    def forward(self, local_feat, global_feat):
        """
        local_feat: (B, C_local, T)
        global_feat: (B, C_global)
        """
        B, C_local, T = local_feat.shape
        B2, C_global = global_feat.shape
        assert B == B2

        # reshape for conv2d
        local_in = local_feat.unsqueeze(-1)        # (B, C_local, T, 1)
        global_in = global_feat.unsqueeze(-1).unsqueeze(-1)  # (B, C_global, 1, 1)



        L = self.local_net(local_in)   # (B, D, T, 1)

        if self.global_net is not None:
            G = self.global_net(global_in)  # (B, D, 1, 1)
        else:
            # if no projection, but channels != mi_units this is an error
            if C_global != self.mi_units:
                raise RuntimeError("global channels != mi_units but no global_net defined")
            # expand to (B, mi_units, 1, 1)
            G = global_in

        L = L.squeeze(-1)   # (B, D, T)
        G = G.squeeze(-1).squeeze(-1)  # (B, D)
        G = G.unsqueeze(-1)  # (B, D, 1)

        #print(">>> debug: local L mean/std:", L.mean().item(), L.std().item())
        #print(">>> debug: global G mean/std:", G.mean().item(), G.std().item())
        # print(">>> debug: shapes L,G:", L.shape, G.shape)



        # optional: detach global embedding to stabilize (experiment)
        # loss = compute_dim_loss(L, G.detach())
        loss = infonce_loss(L, G, temperature=0.07, large_neg=-1e9)
        # print("DIM loss:", loss.item())
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
    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

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
        self.add_module("dim_module", self.dim_module)
        self.add_module("arc_margin", self.arc_margin)

        self.lambda_dim = lambda_dim       # 权重

    def forward(self, outputs ,label):
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
        global_embed, local_feat, logits = outputs

        device = global_embed.device

        # 保证 dim module 在相同的 GPU
        self.dim_module.to(device)

        # global_embed = model_out["global"]
        # local_feat   = model_out["local"]

        # -------- AAM loss --------
        loss_aam = self.arc_margin(logits, label)

        # -------- DIM loss --------
        # LocalDIM 输入：(local_feat, global_embed)
        loss_dim = self.dim_module(local_feat, global_embed)

        # -------- 总损失 --------
        loss_total = loss_aam + self.lambda_dim * loss_dim

        return loss_total, {
            'AAM_loss': loss_aam.item(),
            'DIM_loss': loss_dim.item(),
            'Total_loss': loss_total.item()
        }
    
    def update(self, margin=0.2):
        self.arc_margin.update(margin)