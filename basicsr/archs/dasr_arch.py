import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math
from utils.registry import ARCH_REGISTRY
from .network_swinir import RSTB

class ActLayer(nn.Module):
    """activation layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='leakyrelu'):
        super(ActLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x*1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        else:
            assert 1==0, 'activation type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, norm_type='gn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'wn':
            self.norm = lambda x: torch.nn.utils.weight_norm(x)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            assert 1 == 0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='silu'):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )
        if in_channel != out_channel:
            self.identity = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        else:
            def identity(tensor):
                return tensor
            self.identity = identity

    def forward(self, input):
        res = self.conv(input)
        out = res + self.identity(input)
        return out

class MHCA(nn.Module):
    def __init__(self, dim,
                 Npatch,
                 num_heads=8,
                 Cn=256,
                 c_dim=256,
                 qkv_bias=True,
                 qk_scale=None,
                 alpha=25,
                 attn_drop=0.,
                 proj_drop=0.,
                 LQ_stage=False,
                 extra_norm=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.psN=Npatch
        self.c_dim=c_dim
        self.Cn=Cn
        self.alpha=alpha

        head_dim = c_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, c_dim, bias=qkv_bias)
        if LQ_stage:
            self.k = nn.Linear(c_dim, c_dim, bias=qkv_bias)
        # self.v = nn.Linear(c_dim, c_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(c_dim, c_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.extra_norm = extra_norm
        if self.extra_norm:
            self.normScalar = nn.Parameter(
                torch.ones(num_heads).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))  # [1,num_heads,1,1]

    def build_sinusoidal_embeddings(self,positions: torch.Tensor, embedding_dim: int) -> torch.FloatTensor:
        """
        Based on the implementation in fairseq:
        https://github.com/facebookresearch/fairseq/blob/5ecbbf58d6e80b917340bcbf9d7bdbb539f0f92b/fairseq/modules/sinusoidal_positional_embedding.py#L36
        """
        assert positions.ndim == 2  # [batch, position]--> [B,num_patches]
        half_dim = embedding_dim // 2  # C//4
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=positions.device) * -emb)  # torch.arange(end=half_dim)--> [0,half_dim],step=1  [half_dim]
        emb = positions.unsqueeze(-1) * emb.view(1, 1, -1)  # pos*(1/10000^(2i/d))-->1/10000^(2i/d)=e^{ln[10000^(-2i/d)]}=e^{(-2i/d)*ln(10000)}=e^{(-2i)*[(ln10000)/d]}
        # ↑ [1,B,num_patches] * [1,1,half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # 一半sin，一半cos
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros_like(emb[:, :, :1])], dim=-1)  # 确保pe是偶数的dimension
        return emb

    def create_pe(self,token_embeds,nh,nw,blur=False):
        arange_w = torch.arange(nw)
        arange_h = torch.arange(nh)
        xy_pos = torch.fliplr(torch.cartesian_prod(arange_w,arange_h))
        if blur:
            xy_pos = xy_pos.reshape(nw, nh, -1)
        else:
            xy_pos = xy_pos.unsqueeze(0)  # [1,N, 2]

        batch_size, num_patches, embed_dim = token_embeds.shape  # B, N, C
        if xy_pos.shape[0] == 1:
            xy_pos.expand(batch_size, -1, -1)  # [B,N,2]
        x_pos, y_pos = xy_pos.unbind(dim=-1)  # [B,N]
        x_embeds = self.build_sinusoidal_embeddings(
            x_pos, embed_dim // 2)
        y_embeds = self.build_sinusoidal_embeddings(
            y_pos, embed_dim // 2)
        pos_embeds = torch.cat([x_embeds, y_embeds], dim=-1)  # [B,N,dim]
        return pos_embeds.to(token_embeds.device)

    def forward(self, q_item, v_item, k_item, is_blur=False,nh=16,nw=16):
        B_, N, C = q_item.shape
        Cn, Cdim = v_item.shape

        q = self.q(q_item)
        q = (q + self.create_pe(q,nh=nh,nw=nw)).reshape(B_, N, self.num_heads, Cdim // self.num_heads).permute(0, 2, 1, 3)  # B, num_heads, N, C//num_heads
        q = q * self.scale

        v = v_item.unsqueeze(0)
        v = v.reshape(1, Cn, self.num_heads, Cdim // self.num_heads).permute(0, 2, 1, 3) # B, num_heads, Cn, Cdim//num_heads

        if not is_blur:
            k=k_item
        else:
            k = self.k(k_item)
        k= k.reshape(1, Cn, self.num_heads, Cdim // self.num_heads).permute(0, 2, 1, 3)

        attn_coe = (q @ k.transpose(-2, -1))  # (B,heads,N, Cn)

        attn_coe = self.softmax(attn_coe)
        attn_coe = self.attn_drop(attn_coe)

        x = (attn_coe @ v)  # B, nheads, N, Cdim//num_heads
        if self.extra_norm:
            x = self.normScalar * x
        x = x.transpose(1, 2).reshape(B_, N, Cdim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_coe

class Mlp(nn.Module):
    def __init__(self, ch_in,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 norm_layer=nn.LayerNorm,
                 extra_norm=False):
        super().__init__()
        out_features = out_features or ch_in
        hidden_features = hidden_features or ch_in
        self.fc1 = nn.Linear(ch_in, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.norm = extra_norm
        if self.norm:
            self.norm1 = norm_layer(hidden_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # add norm
        if self.norm:
            x = self.norm1(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
###################################################### Supplementary material: Cross_attention
class MHCA2(nn.Module):
    def __init__(self, dim,
                 Npatch,
                 num_heads=8,
                 Cn=256,
                 c_dim=256,
                 qkv_bias=True,
                 qk_scale=None,
                 alpha=25,
                 attn_drop=0.,
                 proj_drop=0.,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.psN=Npatch
        self.c_dim=c_dim
        self.Cn=Cn
        self.alpha=alpha

        head_dim = c_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, c_dim, bias=qkv_bias)
        self.k = nn.Linear(c_dim, c_dim, bias=qkv_bias)
        self.v = nn.Linear(c_dim, c_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(c_dim, c_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_item, v_item, k_item):
        B_, N, C = q_item.shape
        _,Cn, Cdim = v_item.shape

        q = self.q(q_item)
        q = q.reshape(B_, N, self.num_heads, Cdim // self.num_heads).permute(0, 2, 1, 3)  # B, num_heads, N, C//num_heads
        q = q * self.scale

        k = self.k(k_item)
        k = k.reshape(1, Cn, self.num_heads, Cdim // self.num_heads).permute(0, 2, 1, 3)  # B, num_heads, Cn, Cdim//num_heads

        v = self.v(v_item)
        v = v.reshape(1, Cn, self.num_heads, Cdim // self.num_heads).permute(0, 2, 1, 3) # B, num_heads, Cn, Cdim//num_heads

        attn_coe = (q @ k.transpose(-2, -1))  # (B,heads,N, Cn)

        attn_coe = self.softmax(attn_coe)
        attn_coe = self.attn_drop(attn_coe)

        x= (attn_coe @ v).transpose(1, 2).reshape(B_, N, Cdim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class codebook_transfer(nn.Module):
    def __init__(self, ch_in=64,
                 Cn=256,
                 c_dim=256,
                 Npatch=256,
                 mlp_ratio=4.,
                 num_heads=8,
                 drop=0.,
                 alpha=25,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 mha_depth=1,
                 window_size=8,
                 input_ratio=2,
                 **kwargs):
        super().__init__()

        self.MHA = RSTB(dim=ch_in,
                        input_resolution=(Npatch // input_ratio, Npatch // input_ratio),
                        depth=mha_depth,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        img_size=Npatch,
                        patch_size=1,
                        resi_connection='1conv'
                        )

        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(ch_in)
        self.attn = MHCA2(dim=ch_in,Npatch=Npatch, c_dim=c_dim,Cn=Cn, num_heads=num_heads,alpha=alpha,**kwargs)

        if ch_in != c_dim:
            self.identity1 = nn.Linear(ch_in, c_dim, bias=True)  # ResidualBlock(ch_in, c_dim)
            self.identity2 = nn.Linear(c_dim, ch_in, bias=True)  # ResidualBlock(c_dim, ch_in)
        else:
            def identity(tensor):
                return tensor

            self.identity1 = identity
            self.identity2 = identity

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(c_dim)

        mlp_hidden_dim = int(c_dim * mlp_ratio)
        self.mlp = Mlp(ch_in=c_dim, hidden_features=mlp_hidden_dim, out_features=ch_in, act_layer=act_layer, drop=drop)

    def forward(self, x, c):
        Cn,Cdim=c.shape
        ch, cw = reshape_codebooks(c.shape[-1])
        c = c.unsqueeze(0)  # 1,Cn,Cdim
        x = x.unsqueeze(0)
        x = self.MHA(x, (ch, cw))

        shortcut = x
        # attn
        x = self.norm1(x)
        x = self.attn(x, c,c)
        # FFN
        x = self.identity1(shortcut) + self.drop_path(x)
        x = self.identity2(x) + self.drop_path(self.mlp(self.norm2(x)))
        # attn_coe = self.identity(attn_coe)
        x=x.reshape(Cn,Cdim)
        return x
###################################################### Supplementary material: Cross_attention

###################################################### Supplementary material: DASR
class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(channels_in, channels_in, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(channels_in, self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = nn.Conv2d(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, c,deg):
        '''
        :param x[0]: feature map: 1,Cn,C1,C2    B * C * H * W
        :param x[1]: degradation representation: Cn* C
        '''
        b1, c1, h1, w1 = c.size()

        # branch 1
        kernel = self.kernel(deg).view(-1, 1, self.kernel_size, self.kernel_size)
        # out = self.relu(F.conv2d(c.view(1, -1, h1, w1), kernel, groups=b1*c1, padding=(self.kernel_size-1)//2))
        out = self.relu(F.conv2d(c.view(1, -1, h1, w1), kernel, groups=b1 * c1, padding=(self.kernel_size - 1) // 2))
        out = self.conv(out.view(b1, -1, h1, w1))

        # branch 2
        out = out + self.ca(deg)

        return out

class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, deg):
        '''
        :param x[0]: feature map: 1,Cn,C1,C2   B * C * H * W
        :param x[1]: degradation representation: Cn * C
        '''
        Cn,Cdim=deg.shape
        ch, cw = reshape_codebooks(deg.shape[-1])
        deg = deg.reshape(1, Cn, ch, cw)  # Cn,1,ch,cw

        att = self.conv_du(deg)

        return deg * att

def reshape_codebooks(codes_dim):
    if codes_dim == 512:
        return int(16),int(32)
    elif codes_dim == 128:
        return int(8),int(16)
    ch=math.sqrt(codes_dim)
    return int(ch),int(ch)

class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size,padding=(kernel_size//2))
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size,padding=(kernel_size//2))

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, c,deg):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(c,deg))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2(out, deg))
        out = self.conv2(out) + c

        return out
###################################################### Supplementary material: DASR

def mask_out(x,mask,topk_num):
    topk_v, topk_idx = torch.topk(mask, k=topk_num)

    x = x.view(-1, x.shape[-1])  # B,N,C-->B*N,C
    x = x[topk_idx, :]  # topk_num,C

    return x

def modulate(x, shift, scale,mode=1,mask=None):
    if mode==1:
        shift = shift.view(-1, shift.shape[-1])
        shift = shift[torch.randperm(shift.shape[0])]
        shift = shift[:x.shape[0], :]

        scale = scale.view(-1, scale.shape[-1])
        scale = scale[torch.randperm(scale.shape[0])]
        scale = scale[:x.shape[0], :]
    elif mode==3:
        topk_v,topk_idx=torch.topk(mask,k=x.shape[0])

        shift = shift.view(-1, shift.shape[-1])
        shift = shift[topk_idx,:]

        scale = scale.view(-1, scale.shape[-1])
        scale = scale[topk_idx,:]

    return x * (1 + scale) + shift

class TRG_wrapper(nn.Module):
    def __init__(self, ch_in=64,
                 Cn=256,
                 c_dim=256,
                 Npatch=256,
                 mlp_ratio=4.,
                 num_heads=8,
                 alpha=25,
                 mha_depth=4,
                 window_size=8,
                 input_ratio=2,
                 LR_feat=32,
                 mode=3,
                 **kwargs):
        super().__init__()
        total_layers=4
        self.lq_layer=total_layers-mha_depth
        self.MHA = RSTB(dim=ch_in,
                        input_resolution=(Npatch // input_ratio, Npatch // input_ratio),
                        depth=mha_depth,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        img_size=Npatch,
                        patch_size=1,
                        resi_connection='1conv'
                        )
        TRG_list=nn.ModuleList()
        for _ in range(self.lq_layer):
            TRG_list.append(TRG(ch_in=ch_in, Cn=Cn, c_dim=c_dim,  Npatch=Npatch,
                            mlp_ratio=mlp_ratio, num_heads=num_heads, alpha=alpha,window_size=window_size,input_ratio=input_ratio,LR_feat=LR_feat,mode=mode))
        self.lqag=nn.Sequential(*TRG_list)

    def forward(self, x, c, LR, nh, nw, mode=1):
        x=self.MHA(x,(nh,nw))
        for idx,blk in enumerate(self.lqag):
            x, L_x=blk(x,c,LR,nh,nw,mode)
        if self.lq_layer != 0:
            return x,L_x
        return x,None

class TRG(nn.Module):
    def __init__(self, ch_in=64,
                 Cn=256,
                 c_dim=256,
                 Npatch=256,
                 mlp_ratio=4.,
                 num_heads=8,
                 drop=0.,
                 alpha=25,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 mha_depth=4,
                 window_size=8,
                 input_ratio=2,
                 LR_feat=32,
                 mode=3,
                 extra_norm=False,
                 **kwargs):
        super().__init__()

        self.MHA = RSTB(dim=ch_in,
                        input_resolution=(Npatch // input_ratio, Npatch // input_ratio),
                        depth=mha_depth,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        img_size=Npatch,
                        patch_size=1,
                        resi_connection='1conv'
                        )

        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(ch_in)
        # TRB
        self.attn = MHCA(dim=ch_in,Npatch=Npatch,Cn=Cn, c_dim=c_dim, num_heads=num_heads, LQ_stage=True,alpha=alpha,extra_norm=extra_norm,**kwargs)

        if ch_in != c_dim:
            self.identity1 = nn.Linear(ch_in, c_dim, bias=True)  # ResidualBlock(ch_in, c_dim)
            self.identity2 = nn.Linear(c_dim, ch_in, bias=True)  # ResidualBlock(c_dim, ch_in)
        else:
            def identity(tensor):
                return tensor

            self.identity1 = identity
            self.identity2 = identity

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(c_dim)

        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm3 = norm_layer(c_dim)

        mlp_hidden_dim = int(c_dim * mlp_ratio)
        self.mlp = Mlp(ch_in=c_dim,
                       hidden_features=mlp_hidden_dim,
                       out_features=ch_in,
                       act_layer=act_layer,
                       drop=drop,
                       extra_norm=extra_norm,
                       norm_layer=norm_layer)

        if mode==3: # codebook transformation module
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ch_in + LR_feat, 2 * c_dim+1, bias=True)
            )
        elif mode==4: # Supplementary material: DASR
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ch_in + LR_feat, 1, bias=True)
            ) # B,N,1
            self.DAB=DAB(c_dim)
            self.pre_deg = nn.Sequential(*[Mlp(ch_in=ch_in+LR_feat, out_features=c_dim), Mlp(ch_in=c_dim, out_features=c_dim)])
        elif mode==5: # Supplementary material: Cross_attention
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ch_in + LR_feat, 1, bias=True)
            )  # B,N,1
            self.pre_deg = nn.Sequential(
                *[Mlp(ch_in=ch_in + LR_feat, out_features=c_dim), Mlp(ch_in=c_dim, out_features=c_dim)])
            self.codebook_transfer=codebook_transfer(ch_in=ch_in, Cn=Cn, c_dim=c_dim,  Npatch=Npatch,mlp_ratio=mlp_ratio, num_heads=num_heads, mha_depth=1,window_size=window_size,input_ratio=input_ratio)

        self.LR_down=nn.Conv2d(in_channels=LR_feat, out_channels=LR_feat, kernel_size=3, stride=2, padding=1)

        self.predict_D = nn.Sequential(*[Mlp(ch_in=c_dim, out_features=c_dim),Mlp(ch_in=c_dim, out_features=c_dim)])

    def forward(self, x, c, LR, nh,nw,mode=1):
        x=self.MHA(x,(nh,nw))

        shortcut = x
        LR1=self.LR_down(LR).flatten(2).permute(0,2,1)
        if mode==3: # codebook transformation module
            temp=self.adaLN_modulation(torch.cat((x, LR1), dim=-1))
            mask=temp[:,:,-1].reshape(-1)
            shift_msa, scale_msa = temp[:,:,:-1].chunk(2, dim=-1)
        elif mode==4: # Supplementary material: DASR
            Cn,Cdim=c.shape
            deg1 = torch.cat((x, LR1), dim=-1)
            mask = self.adaLN_modulation(deg1).reshape(-1)

            deg1 = self.pre_deg(deg1)

            deg1 = mask_out(deg1, mask, c.shape[0])  # Cn,C

            ch, cw = reshape_codebooks(c.shape[-1])
            c = c.reshape(1,Cn,ch, cw)  # Cn,1,ch,cw

            d=self.DAB(c,deg1)
        elif mode==5: # Supplementary material: Cross_attention
            Cn, Cdim = c.shape
            deg1 = torch.cat((x, LR1), dim=-1)
            mask = self.adaLN_modulation(deg1).reshape(-1)

            deg1 = self.pre_deg(deg1)
            deg1 = mask_out(deg1, mask, c.shape[0])  # Cn,C
            d=self.codebook_transfer(deg1,c)

        # attn
        x = self.norm1(x)
        if mode==1:
            d = modulate(self.predict_D(c),shift_msa, scale_msa,mode)
            x, L_x = self.attn(x, c, d, is_blur=True,nh=nh,nw=nw)
        elif mode==4:
            d=d.reshape(Cn,Cdim)
            c=c.reshape(Cn,Cdim)
            x, L_x = self.attn(x, c, d, is_blur=True, nh=nh, nw=nw)
        elif mode==5:
            x, L_x = self.attn(x, c, d, is_blur=True, nh=nh, nw=nw)
        elif mode==3:
            d = modulate(self.predict_D(c), shift_msa, scale_msa, mode,mask=mask)
            x, L_x = self.attn(x, c, d, is_blur=True, nh=nh, nw=nw)
            if self.extra_norm:
                x = self.norm3(x)
        else:
            d = self.predict_D(c)
            x, L_x = self.attn(x, c, d, is_blur=True, nh=nh, nw=nw)
            x = modulate(x,shift_msa, scale_msa,mode)
        # FFN
        x = self.identity1(shortcut) + self.drop_path(x)
        x = self.identity2(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x, L_x

class TAG(nn.Module):
    def __init__(self, ch_in=64,
                 Cn=256,
                 c_dim=256,
                 Npatch=256,
                 mlp_ratio=4.,
                 num_heads=8,
                 drop=0.,
                 alpha=25,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 mha_depth=4,
                 window_size=8,
                 input_ratio=2,
                 extra_norm=False,
                 **kwargs):
        super().__init__()

        self.MHA = RSTB(dim=ch_in,
                        input_resolution=(Npatch // input_ratio, Npatch // input_ratio),
                        depth=mha_depth,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        img_size=Npatch,
                        patch_size=1,
                        resi_connection='1conv'
                        )

        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(ch_in)
        # TAB
        self.attn = MHCA(dim=ch_in,Npatch=Npatch, c_dim=c_dim,Cn=Cn, num_heads=num_heads,alpha=alpha,extra_norm=extra_norm,**kwargs)

        if ch_in != c_dim:
            self.identity1 = nn.Linear(ch_in, c_dim, bias=True)  # ResidualBlock(ch_in, c_dim)
            self.identity2 = nn.Linear(c_dim, ch_in, bias=True)  # ResidualBlock(c_dim, ch_in)
        else:
            def identity(tensor):
                return tensor

            self.identity1 = identity
            self.identity2 = identity

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(c_dim)

        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm3 = norm_layer(c_dim)

        mlp_hidden_dim = int(c_dim * mlp_ratio)
        self.mlp = Mlp(ch_in=c_dim, hidden_features=mlp_hidden_dim, out_features=ch_in, act_layer=act_layer, drop=drop,extra_norm=extra_norm,norm_layer=norm_layer)

    def forward(self, x, c,nh,nw):
        x=self.MHA(x,(nh,nw))

        shortcut = x
        # attn
        x = self.norm1(x)
        x, attn_coe = self.attn(x, c,c, is_blur=False,nh=nh,nw=nw)
        if self.extra_norm:
            x = self.norm3(x)
        # FFN
        x = self.identity1(shortcut) + self.drop_path(x)
        x = self.identity2(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_coe

@ARCH_REGISTRY.register()
class DaSR(nn.Module):
    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 upscale=4,
                 codebook_n=64,
                 codebook_dim=64,
                 max_down_ratio=8,
                 channel_list=[ 32, 64, 128,128],
                 nTRG=2,
                 nTAG=3,
                 nMHA=1,
                 d_MHA=4,
                 window_size=8,
                 mlp_ratio=4.,
                 num_heads=8,
                 psN=256,
                 LQ_stage=False,
                 dropout_r=0.0,
                 mode=1,
                 extra_norm=False,
                 **kwargs):
        super(DaSR, self).__init__()
        self.codebook_n = codebook_n
        self.codebook_dim = codebook_dim
        self.blurbook_n = blurbook_n
        self.blurbook_dim = blurbook_dim
        self.upscale = upscale if LQ_stage else 1
        self.nTRG = nTRG
        self.nTAG = nTAG
        self.LQ_stage = LQ_stage
        self.trg_mode = mode
        self.extra_norm = extra_norm

        self.max_down_depth = int(np.log2(max_down_ratio))
        self.encoder_depth = int(np.log2(max_down_ratio // self.upscale))

        self.conv_in = nn.Conv2d(in_channels=in_nc, out_channels=channel_list[0], kernel_size=3, stride=1, padding=1)

        pre_blocks = nn.ModuleList() #LQ
        post_blocks = nn.ModuleList()

        for i in range(self.encoder_depth):
            down_layer = []
            down_layer.append(
                nn.Conv2d(in_channels=channel_list[i], out_channels=channel_list[i + 1], kernel_size=3, stride=2,padding=1))
            down_layer.append(ResidualBlock(channel_list[i + 1], channel_list[i + 1]))
            down_layer.append(ResidualBlock(channel_list[i + 1], channel_list[i + 1]))
            pre_blocks.append(nn.Sequential(*down_layer))
            if i == self.encoder_depth - 1:
                pre_blocks.append(ResidualBlock(channel_list[i + 1], self.codebook_dim)) # 直接映射到码本的维度

        for i in range(self.max_down_depth):
            idx = len(channel_list) - i
            up_layer = []
            if idx ==  len(channel_list):
                up_layer.append(ResidualBlock(self.codebook_dim,channel_list[-1]))
            up_layer.append(nn.Upsample(scale_factor=2))
            up_layer.append(nn.Conv2d(in_channels=channel_list[idx - 1], out_channels=channel_list[idx - 2], kernel_size=3,stride=1, padding=1))
            up_layer.append(ResidualBlock(channel_list[idx - 2], channel_list[idx - 2]))
            up_layer.append(ResidualBlock(channel_list[idx - 2], channel_list[idx - 2]))
            post_blocks.append(nn.Sequential(*up_layer))

        self.encoder_blocks = nn.Sequential(*pre_blocks)
        self.decoder_blocks = nn.Sequential(*post_blocks)
        self.dropout = nn.Dropout2d(p=dropout_r)
        self.out = nn.Conv2d(in_channels=channel_list[0], out_channels=out_nc, kernel_size=3, stride=1, padding=1)

        TAG_list = []
        for i in range(nTAG):
            TAG_list.append(TAG(ch_in=self.codebook_dim,
                                        Cn=codebook_n,
                                        c_dim=codebook_dim,
                                        Npatch=psN,
                                        mlp_ratio=mlp_ratio,
                                        num_heads=num_heads,
                                        mha_depth=d_MHA-1,
                                        window_size=window_size,
                                        input_ratio=(max_down_ratio // self.upscale),
                                        extra_norm=extra_norm,
                                        **kwargs))
        self.TAG_list = nn.Sequential(*TAG_list)

        mha=[]
        for i_layer in range(nMHA):
            layer = RSTB(dim=channel_list[-1],
                         input_resolution=(psN//(max_down_ratio // self.upscale),
                                           psN//(max_down_ratio // self.upscale)),
                         depth=d_MHA,
                         num_heads=num_heads,
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         img_size=psN,
                         patch_size=1,
                         resi_connection='1conv'
                         )
            mha.append(layer)
        self.MHA=nn.Sequential(*mha)

        # Codebook C
        self.codebook = nn.Embedding(codebook_n, codebook_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_n, 1.0 / codebook_dim)

        if self.LQ_stage:
            # TRG
            TRG_list = []
            for i in range(nTRG):
                TRG_list.append(TRG_wrapper(ch_in=self.codebook_dim,
                                Cn=codebook_n,
                                c_dim=codebook_dim,
                                Npatch=psN,mlp_ratio=mlp_ratio,
                                num_heads=num_heads,
                                mha_depth=d_MHA-1,
                                window_size=window_size,
                                input_ratio=(max_down_ratio // self.upscale),
                                LR_feat=channel_list[0],
                                mode=mode,
                                extra_norm=extra_norm,
                                **kwargs))
            self.TRG_list = nn.Sequential(*TRG_list)

    def forward(self, x):
        x = self.conv_in(x)
        LR = x
        x = self.encoder_blocks(x)
        embeddings = [x.clone()]

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # B,H*W,C

        attn_list = []
        if self.LQ_stage:
            for trg in self.TRG_list:
                x, trg_coe = trg(x, self.codebook.weight,LR, nh=H, nw=W,mode=self.trg_mode)
                attn_list.append(trg_coe)

        for tag in self.TAG_list:
            x, tag_coe = tag(x, self.codebook.weight, nh=H, nw=W)
            attn_list.append(tag_coe)

        for layer in self.MHA:
            x=layer(x,(H,W))

        sr_feature = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B,C,H,W

        if not self.LQ_stage:
            embeddings.append(sr_feature.clone())

        for idx, blk in enumerate(self.decoder_blocks):
            sr_feature = blk(sr_feature)

        sr_feature=self.dropout(sr_feature)
        sr_result = self.out(sr_feature)

        if not self.LQ_stage:
            return sr_result, embeddings, attn_list, None
        return sr_result, attn_list, None, embeddings

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.upscale
        output_width = width * self.upscale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile, _ = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.upscale
                output_end_x = input_end_x * self.upscale
                output_start_y = input_start_y * self.upscale
                output_end_y = input_end_y * self.upscale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.upscale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.upscale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.upscale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.upscale

                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[:, :,output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        return output, _

    @torch.no_grad()
    def test(self, input):

        wsz = 8 // self.upscale * 16
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old

        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        dec, _, _ ,  _ = self.forward(input)
        dec = dec[..., :h_old * self.upscale, :w_old * self.upscale]

        return dec, _
