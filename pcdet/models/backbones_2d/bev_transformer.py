import torch
import torch.nn as nn
import numpy as np
from .network_blocks import Focus, SPPBottleneck

class DropPath(nn.Module):
    """DropPath class"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = torch.tensor(keep_prob, dtype=torch.float32)
        keep_prob = keep_prob.cuda()
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=torch.float32).cuda()
        random_tensor = random_tensor.floor()
        output        = inputs.divide(keep_prob) * random_tensor
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='relu6', drop=0):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        if act_layer == 'relu6':
            self.act = nn.ReLU6()
        elif act_layer == 'lrelu':
            self.act = nn.LeakyReLU()
        elif act_layer == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.drop(self.fc2(x))
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, dropout = 0.):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels, out_channels, 2, 2)
        position = torch.randn([1, out_channels, image_size[0] // 2, image_size[1] // 2], requires_grad=True)
        cls      = torch.zeros([1, out_channels, image_size[0] // 2, image_size[1] // 2], requires_grad=True)
        self.position_embedding = nn.Parameter(position)
        self.cls_token          = nn.Parameter(cls)
        self.dropout            = nn.Dropout(dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat([cls_token, x], 1)
        embeddings = x + self.position_embedding
        embeddings = self.dropout(embeddings)
        return embeddings



class Attention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim  = dim
        head_dim  = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.to_qkv    = nn.Conv2d(dim, dim*3, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(attn_drop)

    def rearrange_front(self, inputs, batch_size, height, width):
        features = torch.reshape(inputs, [batch_size, 3, self.dim, height, width])
        features = features.permute([1, 0, 2, 3, 4])
        q, k, v  = features
        return q, k, v


    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = self.rearrange_front(qkv, b, h, w)
        dots    = (q @ k.transpose(-2, -1)) * self.scale
        attn    = dots.softmax(dim=-1)
        out     = attn @ v
        out     = self.proj_drop(self.proj(out))
        return out


class TransBlock(nn.Module):
    def __init__(self, dim, out_dim, num_heads, qk_scale=None, drop=0., act='gelu'):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn  = Attention(dim, num_heads, drop, qk_scale)
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, groups = dim, bias = True)
        self.drop  = DropPath(drop)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * 4)
        self.mlp   = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act, drop=drop)
        self.norm3 = nn.BatchNorm2d(out_dim)


    def forward(self, inputs):
        x = inputs + self.drop(self.attn(self.norm1(inputs)))
        x = x + self.local(self.norm2(x))
        x = x + self.drop(self.mlp(self.norm3(x)))
        return x






class TransBEVBackbone(nn.Module):
    '''
        This class add a transformer layer for BEV backbone to extend it's receptive field.
        We hope the user don't change the basic structure of the 2d backbone to let the code as clean as possible.
    '''
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        
        dim = input_channels
        out_dim   = dim
        num_head  = self.model_cfg.NUM_HEADS
        drop      = self.model_cfg.DROP_RATE
        act       = self.model_cfg.ACT
        num_level = len(layer_nums)
        self.transformer = TransBlock(dim, out_dim, num_head, None, drop, act)
        c_in_list = [input_channels, *num_filters]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_level):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.SiLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.SiLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.SiLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.SiLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_level:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.SiLU(),
            ))

        self.num_bev_features = c_in



    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        spatial_features = self.transformer(spatial_features)
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

class TransSSFA(nn.Module):
    '''
        CIA-SSD version 2d backbone
    '''
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim    = dim // 2
        self.focus = Focus(3, dim)
        self.spp   = SPPBottleneck(dim, dim)
        self.compress = nn.Conv2d(dim * 2, dim, 1, 1)
        num_head  = self.model_cfg.NUM_HEADS
        drop      = self.model_cfg.DROP_RATE
        act       = self.model_cfg.ACT
        self.transformer = TransBlock(dim, out_dim, num_head, None, drop, act)
        self.bottom_up_block_0 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, stride=1, bias=False),
            nn.BatchNorm(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),
        )

        self.bottom_up_block_1 = nn.Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm(256),
            nn.ReLU(),

        )

        self.trans_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),
        )

        self.trans_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),
        )

        self.deconv_block_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),
        )

        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),
        )

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),
        )

        self.w_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm(128),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm(128),
            nn.ReLU(),
        )

        self.w_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm(128),
        )




    def forward(self, data_dict):
        x = data_dict["spatial_features_2d"]
        bev = data_dict["bev"]
        bev_pred = self.spp(self.focus(bev))
        x   = torch.cat([x, bev_pred], 1)
        x   = self.compress(x)
        x   = self.transformer(x)
        x_0 = self.bottom_up_block_0(x)
        x_1 = self.bottom_up_block_1(x_0)
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


class TransBEVNet(nn.Module):
    '''
        SA-SSD Version 2d backbone
    '''
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        num_filters = self.model_cfg.NUM_FILTERS
        dim = input_channels
        out_dim   = dim
        num_head  = self.model_cfg.NUM_HEADS
        drop      = self.model_cfg.DROP_RATE
        act       = self.model_cfg.ACT
        self.fcous    = Focus(3, dim//2)
        self.spp      = SPPBottleneck(dim//2, out_dim)
        self.compress = nn.Conv2d(dim + out_dim, out_dim, 1, 1)
        self.transformer = TransBlock(dim, out_dim, num_head, None, drop, act)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 1),
            nn.BatchNorm2d(num_features),
            nn.SiLU(inplace=True)
        )

    def forward(self, data_dict):
        origin_bev = data_dict["bev"]
        features   = data_dict["spatial_features_2d"]
        origin_for = self.spp(self.fcous(origin_bev))
        origin_for = origin_for.permute(0, 1, 3, 2)
        concat_fea = torch.cat([features, origin_for], 1)
        x = self.compress(concat_fea)
        trans_out  = self.transformer(x)
        result     = self.layers(trans_out)
        data_dict["spatial_features_2d"] = result
        return data_dict

