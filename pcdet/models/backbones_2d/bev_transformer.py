import torch
import torch.nn as nn
import numpy as np

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