from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


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


class SparseXTBlock(spconv.SparseModule):
    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.dw_conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=(3, 9, 9), stride=stride, groups=planes, padding=(4, 4, 1), bias=bias, indice_key=indice_key
        )
        self.ln1      = nn.LayerNorm(planes)
        self.pwconv1  = nn.Linear(planes, planes * 4)
        self.act      = nn.GELU()
        self.pwconv2  = nn.Linear(planes * 4, planes)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        
        out      = self.dw_conv1(x)
        out      = replace_feature(out, self.ln1(out.features))
        out      = replace_feature(out, self.pwconv1(out.features))
        out      = replace_feature(out, self.act(out.features))
        out      = replace_feature(out, self.pwconv2(out.features))
        if self.gamma is not None:
            out  = replace_feature(out, out.features * self.gamma)
        out      = replace_feature(out, self.drop_path(out.features))
        out      = replace_feature(out, out.features + identity.features)
        return out



class VoxelNeXT(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseXTBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 2, norm_fn=norm_fn, stride=2, padding=0, indice_key='spconv2', conv_type='spconv'),
            SparseXTBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 2, norm_fn=norm_fn, stride=2, padding=0, indice_key='spconv3', conv_type='spconv'),
            SparseXTBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 2, norm_fn=norm_fn, stride=2, padding=0, indice_key='spconv4', conv_type='spconv'),
            SparseXTBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(128, 128, (2, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict