import os
import copy
import torch
import torch.nn as nn
import numpy as np

from .network_blocks import Focus, SPPBottleneck, BaseConv, CoordAtt
from .bev_transformer import DropPath, TransBlock
from .swin import BasicLayer



class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PoolFormerLayer(nn.Module):
    def __init__(self, dim, pool_size, depth, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0.):
        super().__init__()
        self.dim    = dim
        self.depth  = depth
        self.blocks = nn.ModuleList([
            PoolFormerBlock(dim, pool_size, mlp_ratio, act_layer, norm_layer, drop, drop_path)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, dropout = 0.):
        super().__init__()
        # down sample BEV image from 704, 800 to 176, 200
        self.patch_embedding = nn.Sequential(
            BaseConv(in_channels, out_channels // 8, 3, 2),
            BaseConv(out_channels // 8, out_channels // 8, 3, 1),
            BaseConv(out_channels // 8, out_channels // 8, 3, 2),
            BaseConv(out_channels // 8, out_channels // 4, 3, 1),
            BaseConv(out_channels // 4, out_channels, 1, 1)
        )

        self.compress        = nn.Conv2d(out_channels * 2, out_channels, 1, 1)
        # To output shape
        position = torch.randn([1, out_channels, image_size[1] // 4, image_size[0] // 4], requires_grad=True)
        # To input shape
        cls      = torch.zeros([1, out_channels, image_size[0] // 4, image_size[1] // 4], requires_grad=True)
        self.position_embedding = nn.Parameter(position)
        self.cls_token          = nn.Parameter(cls)
        self.dropout            = nn.Dropout(dropout)

    def forward(self, batch_data):
        x = batch_data['bev']
        inputs    = batch_data['spatial_features']
        b,c,h,w   = inputs.shape
        cls_token = self.cls_token.expand(x.shape[0], -1, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat([cls_token, x], 1)
        x = self.compress(x)
        x = x.permute(0, 1, 3, 2)
        positions  = x + self.position_embedding
        embeddings = inputs + positions
        embeddings = self.dropout(embeddings)
        return embeddings

class PositionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, dropout = 0):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            BaseConv(in_channels, out_channels // 8, 3, 2),
            BaseConv(out_channels // 8, out_channels // 8, 3, 1),
            BaseConv(out_channels // 8, out_channels // 4, 3, 2),
            BaseConv(out_channels // 4, out_channels // 2, 3, 1),
            BaseConv(out_channels // 2, out_channels, 1, 1)
        )
        self.compress  = BaseConv(out_channels * 2, out_channels, 1, 1)
        self.drop_path = DropPath(dropout)

    def forward(self, batch_data):
        bev_images  = batch_data['bev']
        bev_feature = batch_data['spatial_features']
        position    = self.patch_embedding(bev_images).permute(0, 1, 3, 2)
        cat_feature = torch.cat([bev_feature, position], 1)
        bev_feature = self.drop_path(self.compress(cat_feature))
        return bev_feature


class FourierEmbedding(nn.Module):
    def __init__(self,  in_channels, out_channels, dropout = 0):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            BaseConv(in_channels * 2, out_channels // 4, 3, 2),
            BaseConv(out_channels // 4, out_channels // 4, 3, 1),
            BaseConv(out_channels // 4, out_channels // 2, 3, 2),
            BaseConv(out_channels // 2, out_channels // 2, 3, 1),
            BaseConv(out_channels // 2, out_channels, 1, 1)
        )
        self.compress  = BaseConv(out_channels * 2, out_channels, 1, 1)
        self.drop_path = DropPath(dropout)

    def forward(self, batch_data):
        bev_images   = batch_data['bev']
        bev_feature  = batch_data['spatial_features']
        fourier_bev_x = torch.cos(bev_images * 2 * np.pi)

class TransSPFANet(nn.Module):
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
        self.num_bev_features = 128
        self.num_filters = 256
        self.fcous    = Focus(3, 256)
        self.spp      = SPPBottleneck(256, 256)
        self.compress = nn.Sequential(
            BaseConv(dim + 256, dim, 1, 1),
            BaseConv(dim, dim // 2, 1, 1)
        )
        self.transformer = TransBlock(dim // 2, out_dim // 2, num_head, None, drop, act)
        self.layer_block1 = PoolFormerLayer(128, 7, 3, mlp_ratio=1)
        self.down_sample = BaseConv(128, 128, 3, 2)
        self.layer_block2 = BasicLayer(128, (100, 88), 3, 4, 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        self.weight_spatil = nn.Sequential(
            BaseConv(128, 128, 3, 1),
            BaseConv(128, 1, 1, 1),
        )
        self.weight_segment= nn.Sequential(
            BaseConv(128, 128, 3, 1),
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


        # spatial information group use the poolformer
        block1     = self.layer_block1(trans_out)
        down_block1= self.down_sample(block1)

        # segmation information group use the swin-transformer
        block_temp = down_block1.permute(0, 2, 3, 1)
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


class TransSPoolformer(nn.Module):
    '''
        CIA-SSD version 2d backbone
    '''
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim    = dim
        '''self.position_embedding = nn.Sequential(
            BaseConv(3, 64, 4, 4),
            BaseConv(64, 128, 1, 1),
            BaseConv(128, 256, 1, 1),
        )'''
        self.position_embedding = PatchEmbedding(3, 256, (704, 800))
        self.project     = nn.Conv2d(out_dim, out_dim // 2, 1)
        self.bottom_up_block_0 = nn.Sequential(
            BaseConv(128, 128, 3, 1),
            BaseConv(128, 128, 3, 1),
            BaseConv(128, 128, 3, 1),
        )
        self.num_bev_features = 128

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
 
        x = self.position_embedding(data_dict)

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

class TransSwinBase(nn.Module):
    '''
        CIA-SSD version 2d backbone
    '''
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim    = dim
        self.position_embedding = nn.Sequential(
            BaseConv(3, 64, 4, 4),
            BaseConv(64, 128, 1, 1),
            BaseConv(128, 256, 1, 1),
        )
        self.project     = nn.Conv2d(out_dim, out_dim // 2, 1)
        self.bottom_up_block_0 = nn.Sequential(
            BaseConv(128, 128, 3, 1),
            BaseConv(128, 128, 3, 1),
            BaseConv(128, 128, 3, 1),
        )
        self.num_bev_features = 128

        self.bottom_up_block_1 = nn.Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.swin_block = BasicLayer(256, (100, 88), 5, 8, 4)


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
        bev = bev.permute(0, 1, 3, 2)
        position = self.position_embedding(bev)
        x   = x + position

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

class Trans_Coor_Swin_Net(nn.Module):
    '''
        Coordinate_SSD
    '''
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim    = dim
        self.position_embedding = PositionEmbedding(3, 256, 0.1)
        self.project     = nn.Conv2d(out_dim, out_dim // 2, 1)
        self.spatial_block = CoordAtt(128, 128, 16)
        self.num_bev_features = 128

        self.bottom_up_block_1 = BaseConv(128, 256, 3, 2)
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
 
        x = self.position_embedding(data_dict)

        x   = self.project(x)
        spatial_mask = self.spatial_block(x)
        x_0          = spatial_mask * x
        
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