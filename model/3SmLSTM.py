import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# sencond-order pooling
from .MPNCOV import CovpoolLayer,SqrtmLayer,TriuvecLayer
# mLSTM
from .vision_lstm_Tconv import ViLBlock
from .vision_lstm_util import combine_X,reverse_X,MultiScale_TemporalConv


### torch version too old for timm
### https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)





def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)



class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), groups=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x



class unit_vit(nn.Module):
    def __init__(self, dim_in, dim,  add_skip_connection=True,  drop=0.,
                 attn_drop=0.,
                 drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer=0,
                 insert_cls_layer=0, pe=False, num_point=25, **kwargs):
        super().__init__()
        # 仅对单词进行归一化，对C维度
        # self.norm1 = norm_layer(dim_in)
        self.dim_in = dim_in
        self.dim = dim
        self.add_skip_connection = add_skip_connection
        self.num_point = num_point
        # self.attn = MHSA(dim_in, dim, A, num_heads=num_of_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                  attn_drop=attn_drop,
        #                  proj_drop=drop, insert_cls_layer=insert_cls_layer, pe=pe, num_point=num_point, layer=layer,
        #                  **kwargs)

        self.lstm = ViLBlock(dim=dim_in, conv_kernel_size=3,conv_kind='Tconv',num_point=num_point)
        # self.lstm = ViLBlock(dim=dim_in, conv_kernel_size=3, A=A)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()



        # 等维度映射
        # self.same_proj = nn.Conv2d(dim, dim, 1, bias=False)
        # self.learnable_skip = nn.Parameter(torch.ones(dim))
        # self.pe = pe

    def forward(self, x):



        # [B*M,C,T,V]
        B, C, T, V = x.shape
        # print('before:', x.shape)

        # x=reverse_X(self.lstm(change_X(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))), T)
        x= reverse_X(self.lstm(combine_X(x),T),T)



        return x


class TCN_ViL_unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=True, kernel_size=5,
                 dilations=[1, 2],  num_point=25, layer=0):
        super(TCN_ViL_unit, self).__init__()
        self.vit1 = unit_vit(in_channels, out_channels, add_skip_connection=residual,
                             num_point=num_point, layer=layer)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            # redisual=True has worse performance in the end
                                            residual=False)
        self.act = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # 这里的残差块如果有stride则要变成时间卷积
        y = self.act(self.tcn1(self.vit1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=20, num_person=2, num_frames=64, graph=None, graph_args=dict(),
                 in_channels=3,
                 drop_out=0.2, dim=144, **kwargs):

        super(Model, self).__init__()

        # if graph is None:
        #     raise ValueError()
        # else:
        #     Graph = import_class(graph)
        #     self.graph = Graph(**graph_args)

        # A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        # self.joint_label = joint_label

        embed_dim = dim
        # 先创建一个模组的list，后面直接传给nn.ModuleList就可以成组了，可以自定义内部模块的顺序，不同于Sequential(不用List)
        # stem = []
        # 由通道数加倍的卷积+激活函数+通道数由2倍到3倍的卷积+激活函数+通道数变为embed_dim的卷积组成
        # 总的就是把通道数翻3倍后再变为embed_dim
        # Conv2d进行卷积时的输入为(batch_size,channels,h,w)所以改变的是第1维


        # 备选一次卷积到目标维度
        self.proj_up = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(1, 1), stride=(1, 1),
                                 padding=(0, 0))
        # 位置编码测试
        # self.pos_emb=Pos_Embed(embed_dim,num_frames,num_point)

        # 可学习的时空编码
        # self.joint_person_temporal_embedding = nn.Parameter(
        #     torch.zeros(1, embed_dim, num_frames, num_point ))
        # trunc_normal_(self.joint_person_temporal_embedding, std=.02)

        self.l1 = TCN_ViL_unit(embed_dim, embed_dim, residual=True,num_point=num_point, layer=1)
        # * num_heads, effect of concatenation following the official implementation
        self.l2 = TCN_ViL_unit(embed_dim, embed_dim, residual=True, num_point=num_point, layer=2)
        self.l3 = TCN_ViL_unit(embed_dim, embed_dim, residual=True,num_point=num_point, layer=3)
        self.l4 = TCN_ViL_unit(embed_dim, embed_dim, residual=True,num_point=num_point, layer=4)
        # 第五层和第8层stide=2使得时间维度减半
        self.l5 = TCN_ViL_unit(embed_dim, embed_dim, residual=True, stride=2, num_point=num_point, layer=5)
        self.l6 = TCN_ViL_unit(embed_dim, embed_dim, residual=True, num_point=num_point, layer=6)
        self.l7 = TCN_ViL_unit(embed_dim, embed_dim, residual=True, num_point=num_point, layer=7)
        self.l8 = TCN_ViL_unit(embed_dim, embed_dim, residual=True, stride=2,num_point=num_point, layer=8)
        self.l9 = TCN_ViL_unit(embed_dim, embed_dim, residual=True,num_point=num_point, layer=9)
        self.l10 = TCN_ViL_unit(embed_dim, embed_dim, residual=True,num_point=num_point, layer=10)



        # standard ce loss
        # self.fc = nn.Linear(embed_dim, num_class)

        # self.isqrt_dim = 144  # 下次换256
        # self.layer_reduce = nn.Conv2d(embed_dim, self.isqrt_dim, kernel_size=1, stride=1, padding=0,
        # bias=False)
        # self.layer_reduce_bn = nn.BatchNorm2d(self.isqrt_dim)
        # self.layer_reduce_relu = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(int(self.isqrt_dim * (self.isqrt_dim + 1)), num_class)
        # self.fc = nn.Linear(int(self.isqrt_dim * (self.isqrt_dim + 1) / 2), num_class)

        self.fc = nn.Linear(int(embed_dim * (embed_dim + 1) / 2), num_class)


        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        ## n, c, t, v
        x = x.view(N, M, V, C, T).contiguous().view(N * M, V, C, T).permute(0, 2, 3, 1)

        # 新增的投影层
        # for layer in self.stem:
        #     # 将in_channels变为embed_dim
        #     x = layer(x)
        x = self.proj_up(x)

        # 投影后加一个pos_embed就行，很简单
        # x=self.pos_emb(x)+x

        # 可学习时空编码
        # x=self.joint_person_temporal_embedding+x

        x = self.l1(x)

        x = self.l2(x)

        x = self.l3(x)

        x = self.l4(x)

        x = self.l5(x)

        x = self.l6(x)

        x = self.l7(x)

        x = self.l8(x)

        x = self.l9(x)

        x = self.l10(x)


        # Vision-lstm里作者在经过所有的lstm后还有个norm


        # second-order pooling
        # x = self.layer_reduce(x)
        # x = self.layer_reduce_bn(x)
        # x = self.layer_reduce_relu(x)

        x = CovpoolLayer(x)
        x = SqrtmLayer(x, 3)
        x = TriuvecLayer(x)

        # print('shape',x.shape)

        _, C  = x.size()
        # print('c',C)
        x = x.view(N, M, -1)
        x = x.mean(1)

        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc(x)

        # # spatial temporal average pooling
        # x = x.view(N, M, C, -1)
        # x = x.mean(3).mean(1)
        # x = self.drop_out(x)
        # x = self.fc(x)

        return x
