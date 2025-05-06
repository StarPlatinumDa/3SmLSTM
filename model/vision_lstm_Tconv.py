# This file is licensed under AGPL-3.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel
import math
import warnings
from enum import Enum

import einops
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from .vision_lstm_util import  DropPath,combine_X,reverse_X,MultiScale_TemporalConv





def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def parallel_stabilized_simple(
        pos_emb: torch.Tensor,

        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,

) -> torch.Tensor:
    """
    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        # 一个下三角矩阵，用于实现因果关系，确保在时间序列中不会出现未来信息的泄露。是一个布尔值矩阵
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
        # 是否按行稳定组合矩阵C（每行取最大值） 替代方案：减去所有行的最大值。默认为True。
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.  避免除0

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape # B,头个数,序列长度，值维度
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix  log(遗忘门ft)
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        # 如果没有提供下三角矩阵或大小不匹配(矩阵比S维度大)
        # 则创建一个默认的下三角矩阵(S,S)
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        # 否则直接用
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            # -2维度依次累加例如1,2,3->1,3,6
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    # 对于每个批次/头，这是一个矩阵形状为（S+1，S+1），其中包含log(遗忘门)的cumsum
    # 在第二维度（column维度）中。每一行都有相同的内容。这是第一行的副本。
    # 每行的第一个条目为零。
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)


    # k-hop
    # print('pos_emb',pos_emb.shape)
    V,M,C=pos_emb.shape

    pe = pos_emb.view(V, V, NH, C // NH)
    # [b,nh,v,c]*[v,v,nh,c]
    k_hop_attn=torch.einsum('bhvc,vmhc->bhvm',queries,pe)


    # combination matrix C
    # print('q*k',queries.shape,keys_scaled.shape) # 头个数4，值维度60 [2048,4,25,60]*[2048,4,60,25]->[2048,4,25,25]
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)

    # print('qk_matrix * D_matrix',qk_matrix.shape, D_matrix.shape)
    # C_matrix = qk_matrix * D_matrix  # (B, NH, S, S) [2048,4,25,25]*[2048,4,25,25]
    C_matrix = (qk_matrix+k_hop_attn) * D_matrix

    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    # print('h_tilde=C_matrix_normalized @ values',C_matrix_normalized.shape,values.shape)
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH) [2048,4,25,25]*[2048,4,25,60]->[2048,4,25,60]

    return h_tilde_state


class LinearHeadwiseExpand(nn.Module):
    """
    先分裂出头个数的维度nh
    然后用准备好的weight与分裂的特征相乘
    即 nh,d * nh,d,out_d得到nh out_d
    就相当于在多个头上分别进行了投影将值维度d投影到out_d
    最后再合并out_d和nh得到总结果

    投影到高维空间
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 维度分出一个num_head
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        # 爱因斯坦求和nh,d * nh,d,out_d得到nh out_d
        # 即分离出头个数 后对每个分组分别将d投影到out_d
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        # 再将新的out_d与头个数合并为1个维度
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            # 再加上bias
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )



class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    """
    分别对每个头进行norm
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class MatrixLSTMCell(nn.Module):
    """
    对应原始的cell
    有各种门，对qkv进行维度处理得到多头，并且加上mask加入因果
    然后将qkv,i,f,mask输入到并行计算中进行最终的计算得到h
    对h进行MultiHeadLayerNorm  分别对每个头进行norm
    返回norm后的结果作为h_tilda
    """
    def __init__(self, dim, num_heads, norm_bias=True,num_point=25):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()


        self.hops = np.zeros((num_point, num_point))
        curnum = 1
        # 总共有(25*25-25)/2=300种联系
        for i in range(num_point):
            for j in range(i + 1, num_point):
                self.hops[i][j] = curnum
                self.hops[j][i] = self.hops[i][j]
                curnum += 1

        self.hops = torch.tensor(self.hops).long()
        #
        self.rpe = nn.Parameter(torch.zeros((curnum, dim)))
        # self.rpe = nn.Parameter(torch.zeros((self.hops.max() + 1, dim)))


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        # # k-hop distance encoding,
        pos_emb = self.rpe[self.hops]

        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        # print('if_gate_input:',if_gate_input.shape)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH) (768,64,4,6)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        # print('q,k,v view',q.shape,k.shape,v.shape)

        q = q.transpose(1, 2)  # (B, NH, S, DH) (768,4,64,6)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)
        # print('q,k,v transpose', q.shape, k.shape, v.shape)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        # print('igate_preact',igate_preact.shape)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        # print('fgate_preact',fgate_preact.shape)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
            pos_emb=pos_emb,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class ViLLayer(nn.Module):
    """
    与原始的xLSTM中mLSTM的layer相同
    首先在上一层的block进行了层归一化
    根据方向选择翻转与否
    这里进行处理得到了qkv,用qkv计算得到h_tilda，直到project_down都与原始的相同
    计算完后再根据翻转与否翻转回去
    """
    def __init__(
            self,
            dim,
            # direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="2d",
            num_point=25,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        # self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind

        # inner_dim = expansion * dim
        inner_dim = 1 * dim

        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        if conv_kind == 'Tconv':
            self.conv =MultiScale_TemporalConv(
                inner_dim, inner_dim, kernel_size=5, stride=1,
                dilations=[1,2],
                residual=False
            )
        else:
            raise NotImplementedError
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
            num_point=num_point
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.conv1 = MultiScale_TemporalConv(
            inner_dim, inner_dim, kernel_size=5, stride=1,
            dilations=[1, 2],
            residual=False
        )



        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor,T) -> torch.Tensor:
        B, S, _ = x.shape

        # up-projection
        # 向上投影将值维度从dim变成2*expansion*dim
        # print('before proj:',x.shape) # [768,64,12]
        x_inner = self.proj_up(x) # [768,64,48]
        # 将值维度分成两个expansion*dim  [B,S,expansion*dim]
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)
        # print('after proj x_mlstm:', x_mlstm.shape)# [768,64,24]

        # mlstm branch  因果1维卷积 或 3维转4维进行2维卷积
        # x_mlstm_conv = self.conv(x_mlstm)

        # 改成时间卷积
        # print('reverse',reverse_X(x_mlstm,T).shape)
        x_mlstm_conv=combine_X(self.conv(reverse_X(x_mlstm, T)))
        # print('x_mlstm_conv',x_mlstm_conv.shape)




        # print('after conv x_mlstm:', x_mlstm_conv.shape) # 分组卷积不改变维度大小 [768,64,24]
        # 激活函数
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        # 得到Q,K
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        # V就是原来没卷积激活过的
        v = self.v_proj(x_mlstm)
        # print('q,k,v:',q.shape,k.shape,v.shape) # 分组投影都不改变维度，q,k,v都为[768,64,24]
        # 用qkv计算h_tilde
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        # print('h_tilde',h_tilde_state.shape)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        z=combine_X(self.conv1(reverse_X(z, T)))
        # output / z branch
        # print('skip,z',h_tilde_state_skip.shape,z.shape)
        # h_state = h_tilde_state_skip * F.silu(z)
        h_state = h_tilde_state_skip + F.silu(z)
        # print('h_state', h_state.shape)

        # down-projection
        x = self.proj_down(h_state)
        # print('final_x',x.shape)


        return x

    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()


class ViLBlock(nn.Module):
    """
    layernorm后经过ViLLayer
    然后drop_path得到每个样本的有效下降路径（随机深度）
    """
    def __init__(
            self,
            dim,
            # direction,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            num_point=25,
    ):
        super().__init__()
        self.dim = dim
        # self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=dim,
            # direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
            num_point=num_point
        )

        # 等维度映射
        # self.same_proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.learnable_skip = nn.Parameter(torch.ones(dim))

        self.reset_parameters()

    def _forward_path(self, x,T):
        x = self.norm(x)
        # x = self.layer(x)
        x = self.layer(x,T) + x * self.learnable_skip
        return x

    def forward(self, x: torch.Tensor,T) -> torch.Tensor:
        # 这里的输入还是embedding后flatten的[128,64,192]   [B,序列长度,dim]
        x = self.drop_path(x, self._forward_path,{'T':T})

        # 不加norm和drop_path
        # x=self._forward_path(x)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()
        nn.init.ones_(self.learnable_skip)


