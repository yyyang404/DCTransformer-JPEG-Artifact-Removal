import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from itertools import repeat
import collections.abc


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f'activation {act_type} is not found')
    return layer


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


"""
Spatial Branch of Swin Transformer
"""


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1, 2).transpose(0, 1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W': output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2),
                                                 dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]),
            device=self.relative_position_params.device,
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type='SW', input_resolution=None):
        """
        SwinTransformer Block
        in/out: tensor with shape of [b h w c]
        """
        super(SwinBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution is not None and input_resolution <= window_size:
            self.type = 'W'
        self.window_size = window_size
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, self.window_size, self.type)
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.msa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


"""
Frequency-wiseAttentionBlock
"""


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FE_MSA(nn.Module):
    def __init__(
            self,
            dim,
            head_dim=64,  # each head's dim
            n_heads=2,  # num of heads
    ):
        super().__init__()
        self.num_heads = n_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, head_dim * n_heads, bias=True)
        self.to_k = nn.Linear(dim, head_dim * n_heads, bias=True)
        self.to_v = nn.Linear(dim, head_dim * n_heads, bias=True)
        self.rescale = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.proj = nn.Linear(head_dim * n_heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
        )  # learnable pos emb
        # self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """

        b, h, w, c = x_in.shape

        pos_emb = Rearrange('b c h w -> b h w c')(self.pos_emb(Rearrange('b h w c -> b c h w')(x_in)))
        x_in = x_in + pos_emb

        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp))
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.head_dim)
        out_c = self.proj(x).view(b, h, w, c)

        return out_c


class FEBlock(nn.Module):
    def __init__(
            self,
            dim,
            head_dim=32,
            ff_mul=4
    ):
        super().__init__()
        n_heads = dim // head_dim
        # n_heads = 4
        self.attn = FE_MSA(dim=dim, head_dim=head_dim, n_heads=n_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff_mul = ff_mul
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * self.ff_mul),
            nn.GELU(),
            nn.Linear(dim * self.ff_mul, dim),
        )

    def forward(self, x):
        """
        in/out: [b,h,w,c]
        """
        x = x + self.attn(self.ln1(x))
        x_out = x + self.mlp(self.ln2(x))

        return x_out


class SpatFreqTransBlock(nn.Module):
    def __init__(
            self,
            dim,
            spat_head_dim=32,  # SwinT: default 32
            freq_head_dim=32,  # MAT: default 64, we set as 32 here.
            swin_type='W',
            win_size=4,
            attn_concat="3conv",
            rescon_attn=True
    ):
        super().__init__()
        assert swin_type in ['W', 'SW']
        self.swin_type = swin_type
        self.win_size = win_size
        self.spat_branch = SwinBlock(input_dim=dim, output_dim=dim,
                                     head_dim=spat_head_dim, window_size=self.win_size,
                                     type=self.swin_type, input_resolution=256)

        self.freq_branch = FEBlock(dim=dim, head_dim=freq_head_dim)

        assert attn_concat in ['1conv', '3conv', '2x3conv']
        if attn_concat == "2x3conv":
            self.concat_conv = nn.Sequential(
                nn.Conv2d(dim * 2, dim * 2, 1, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim, 1, 1, bias=True)
            )
        elif attn_concat == "3conv":
            self.concat_conv = nn.Sequential(nn.Conv2d(dim * 2, dim, 3, 1, bias=True))
        elif attn_concat == "1conv":
            self.concat_conv = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, 1, bias=True))
        self.rescon_attn = rescon_attn

    def forward(self, x):
        """
        x: [b,c,h,w], out: [b,c,h,w]
        """
        x = Rearrange('b c h w -> b h w c')(x)  # x_attn: [b,h,w,c]

        if self.rescon_attn:  # residual connection in each attention blocks
            x_freq = self.freq_branch(x) + x
            x_spat = self.spat_branch(x) + x

            x_freq = Rearrange('b h w c -> b c h w')(x_freq)  # x_freq: [b,c,h,w]
            x_spat = Rearrange('b h w c -> b c h w')(x_spat)  # x_spat: [b,c,h,w]

            x_out = self.concat_conv(torch.cat([x_freq, x_spat], dim=1))

        else:  # residual connection after the concatenate conv
            x_freq = self.freq_branch(x)
            x_spat = self.spat_branch(x)

            x_freq = Rearrange('b h w c -> b c h w')(x_freq)  # x_freq: [b,c,h,w]
            x_spat = Rearrange('b h w c -> b c h w')(x_spat)  # x_spat: [b,c,h,w]

            x_out = self.concat_conv(torch.cat([x_freq, x_spat], dim=1)) + x

        return x_out


class DCTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_blocks=6,
            spat_head_dim=32,  # SwinT: default 32
            freq_head_dim=32,  # MAT: default 64, we set as 32 here.
            win_size=4,
            attn_concat="2x3conv",
            rescon_attn=True,
    ):
        super().__init__()
        self.sfsa_blocks = nn.ModuleList([
            SpatFreqTransBlock(dim=dim,
                               spat_head_dim=spat_head_dim,
                               freq_head_dim=freq_head_dim,
                               swin_type='W' if not i % 2 else 'SW',
                               win_size=win_size,
                               attn_concat=attn_concat,
                               rescon_attn=rescon_attn,
                               )
            for i in range(num_blocks)])
        self.group_tail_conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)

    def forward(self, x):
        for blk in self.sfsa_blocks:
            x = blk(x)
        x = self.group_tail_conv(x)
        return x
