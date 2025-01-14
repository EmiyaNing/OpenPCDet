# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .network_blocks import Focus, SPPBottleneck, BaseConv
from .bev_transformer import DropPath, TransBlock


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.kaiming_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class TransSwinFA(nn.Module):
    '''
        CIA-SSD version 2d backbone
    '''
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim    = dim
        self.focus = Focus(3, 256)
        self.spp   = SPPBottleneck(256, 256)
        self.compress = nn.Conv2d(dim + 256, dim, 1, 1)
        num_head  = self.model_cfg.NUM_HEADS
        drop      = self.model_cfg.DROP_RATE
        act       = self.model_cfg.ACT
        self.num_bev_features = 128
        self.transformer = TransBlock(dim, out_dim, num_head, None, drop, act)
        self.project     = nn.Conv2d(out_dim, out_dim // 2, 1)
        self.bottom_up_block_0 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.bottom_up_block_1 = nn.Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.swin_block = BasicLayer(256, (100, 88), 3, 4, 4)


        self.trans_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.w_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.w_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )




    def forward(self, data_dict):
        x = data_dict["spatial_features"]
        bev = data_dict["bev"]
        bev_pred = self.spp(self.focus(bev))
        bev_pred = bev_pred.permute(0, 1, 3, 2)
        x   = torch.cat([x, bev_pred], 1)
        x   = self.compress(x)
        x   = self.transformer(x)
        x   = self.project(x)
        x_0 = self.bottom_up_block_0(x)
        x_1 = self.bottom_up_block_1(x_0)
        x_1 = x_1.permute(0, 2, 3, 1)
        b, h, w, c = x_1.shape
        x_1 = x_1.reshape(b, h * w, c)
        x_1 = self.swin_block(x_1)
        x_1 = x_1.reshape(b, h, w, c)
        x_1 = x_1.permute(0, 3, 1, 2)
        x_trans_0 = self.trans_0(x_0)
        x_trans_1 = self.trans_1(x_1)
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0
        x_middle_1 = self.deconv_block_1(x_trans_1)
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]
        data_dict["spatial_features_2d"] = x_output.contiguous()

        return data_dict




class TransSwinFAV2(nn.Module):
    '''
        CIA-SSD version 2d backbone
    '''
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim    = dim
        self.focus = Focus(3, 256)
        self.spp   = SPPBottleneck(256, 256)
        self.compress = nn.Conv2d(dim + 256, dim, 1, 1)
        num_head  = self.model_cfg.NUM_HEADS
        drop      = self.model_cfg.DROP_RATE
        act       = self.model_cfg.ACT
        self.num_bev_features = 128
        self.transformer = TransBlock(dim, out_dim, num_head, None, drop, act)
        self.project     = nn.Conv2d(out_dim, out_dim // 2, 1)
        self.bottom_up_block_0 = nn.Sequential(
            BaseConv(128, 128, 3, 1),
            BaseConv(128, 128, 3, 1),
            BaseConv(128, 128, 3, 1),
        )

        self.bottom_up_block_1 = nn.Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.swin_block = BasicLayer(256, (100, 88), 3, 8, 4)


        self.trans_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.w_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.w_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )


    def forward_swin_block_1(self, inputs):
        x = inputs.permute(0, 2, 3, 1)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        x = self.swin_block(x)
        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2)
        return x


    def forward(self, data_dict):
        x = data_dict["spatial_features"]
        bev = data_dict["bev"]
        bev_pred = self.spp(self.focus(bev))
        bev_pred = bev_pred.permute(0, 1, 3, 2)
        x   = torch.cat([x, bev_pred], 1)
        x   = self.compress(x)
        x   = self.transformer(x)
        x   = self.project(x)


        x_0 = self.bottom_up_block_0(x)



        x_1 = self.bottom_up_block_1(x_0)
        x_1 = self.forward_swin_block_1(x_1)

        x_trans_0 = self.trans_0(x_0)
        x_trans_1 = self.trans_1(x_1)
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0
        x_middle_1 = self.deconv_block_1(x_trans_1)
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]
        data_dict["spatial_features_2d"] = x_output.contiguous()

        return data_dict


class TransSWINNet(nn.Module):
    '''
        SWIN with BEV INPUT branch.
    '''
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim   = dim
        num_head  = self.model_cfg.NUM_HEADS
        drop      = self.model_cfg.DROP_RATE
        act       = self.model_cfg.ACT
        self.num_bev_features = 256
        self.num_filters = 256
        self.fcous    = Focus(3, 256)
        self.spp      = SPPBottleneck(256, 256)
        self.compress = nn.Conv2d(dim + 256, dim, 1, 1)
        self.transformer = TransBlock(dim, out_dim, num_head, None, drop, act)
        self.down_sample = BaseConv(256, 256, 3, 2)
        self.layers = BasicLayer(256, (100, 88), 6, 4, 4)
        self.deconv = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)


    def forward(self, data_dict):
        origin_bev = data_dict["bev"]
        features   = data_dict["spatial_features"]
        origin_for = self.spp(self.fcous(origin_bev))
        origin_for = origin_for.permute(0, 1, 3, 2)
        concat_fea = torch.cat([features, origin_for], 1)
        x = self.compress(concat_fea)
        trans_out  = self.transformer(x)
        trans_out  = self.down_sample(trans_out)
        trans_out  = trans_out.permute(0, 2, 3, 1)
        b, h, w, c = trans_out.shape
        trans_out  = trans_out.reshape(b, h * w, c)
        result     = self.layers(trans_out)
        result     = result.reshape(b, h, w, c)
        result     = result.permute(0, 3, 1, 2)
        result     = self.deconv(result)
        data_dict["spatial_features_2d"] = result
        return data_dict


class TransSWINNetV2(nn.Module):
    '''
        SWIN with BEV INPUT branch.
    '''
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim   = dim
        num_head  = self.model_cfg.NUM_HEADS
        drop      = self.model_cfg.DROP_RATE
        act       = self.model_cfg.ACT
        self.num_bev_features = 256
        self.num_filters = 256
        self.fcous    = Focus(3, 256)
        self.spp      = SPPBottleneck(256, 256)
        self.compress = BaseConv(dim + 256, dim, 1, 1)
        self.transformer = TransBlock(dim, out_dim, num_head, None, drop, act)
        self.layer_block1 = BasicLayer(256, (200, 176), 3, 4, 8)
        self.down_sample = BaseConv(256, 256, 3, 2)
        self.layer_block2 = BasicLayer(256, (100, 88), 3, 4, 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )



    def forward(self, data_dict):
        origin_bev = data_dict["bev"]
        features   = data_dict["spatial_features"]
        origin_for = self.spp(self.fcous(origin_bev))
        origin_for = origin_for.permute(0, 1, 3, 2)
        concat_fea = torch.cat([features, origin_for], 1)
        x = self.compress(concat_fea)
        trans_out  = self.transformer(x)

        trans_out  = trans_out.permute(0, 2, 3, 1)
        b, h, w, c = trans_out.shape
        trans_out  = trans_out.reshape(b, h * w, c)
        block1     = self.layer_block1(trans_out)
        block1     = block1.reshape(b, h, w, c)
        block1     = block1.permute(0, 3, 1, 2)
        block1     = self.down_sample(block1)

        block_temp = block1.permute(0, 2, 3, 1)
        b, h, w, c = block_temp.shape
        block_temp = block_temp.reshape(b, h * w, c)
        block2     = self.layer_block2(block_temp)
        block2     = block2.reshape(b, h, w, c)
        block2     = block2.permute(0, 3, 1, 2)
        block2     = self.deconv(block2)

        result     = block1 + block2

        data_dict["spatial_features_2d"] = result
        return data_dict


class TransSWINFFANet(nn.Module):
    '''
        SWIN with BEV INPUT branch.
    '''
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim   = dim
        num_head  = self.model_cfg.NUM_HEADS
        drop      = self.model_cfg.DROP_RATE
        act       = self.model_cfg.ACT
        self.num_bev_features = 256
        self.num_filters = 256
        self.fcous    = Focus(3, 256)
        self.spp      = SPPBottleneck(256, 256)
        self.compress = BaseConv(dim + 256, dim, 1, 1)
        self.transformer = TransBlock(dim, out_dim, num_head, None, drop, act)
        self.layer_block1 = BasicLayer(256, (200, 176), 3, 4, 8)
        self.down_sample = BaseConv(256, 256, 3, 2)
        self.layer_block2 = BasicLayer(256, (100, 88), 3, 4, 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        self.weight_spatil = nn.Sequential(
            BaseConv(256, 128, 3, 1),
            BaseConv(128, 1, 1, 1),
        )
        self.weight_segment= nn.Sequential(
            BaseConv(256, 128, 3, 1),
            BaseConv(128, 1, 1, 1),
        )



    def forward(self, data_dict):
        origin_bev = data_dict["bev"]
        features   = data_dict["spatial_features"]
        origin_for = self.spp(self.fcous(origin_bev))
        origin_for = origin_for.permute(0, 1, 3, 2)
        concat_fea = torch.cat([features, origin_for], 1)
        x = self.compress(concat_fea)
        trans_out  = self.transformer(x)



        trans_out  = trans_out.permute(0, 2, 3, 1)
        b, h, w, c = trans_out.shape
        trans_out  = trans_out.reshape(b, h * w, c)
        block1     = self.layer_block1(trans_out)
        block1     = block1.reshape(b, h, w, c)
        block1     = block1.permute(0, 3, 1, 2)
        block1     = self.down_sample(block1)

        block_temp = block1.permute(0, 2, 3, 1)
        b, h, w, c = block_temp.shape
        block_temp = block_temp.reshape(b, h * w, c)
        block2     = self.layer_block2(block_temp)
        block2     = block2.reshape(b, h, w, c)
        block2     = block2.permute(0, 3, 1, 2)
        block2     = self.deconv(block2)

        weight1    = self.weight_spatil(block1)
        weight2    = self.weight_segment(block2)

        weight     = torch.softmax(torch.cat([weight1, weight2], dim=1), dim=1)

        result     = block1 * weight[:, 0:1, :, :] + block2 * weight[:, 1:2, :, :]

        data_dict["spatial_features_2d"] = result
        return data_dict