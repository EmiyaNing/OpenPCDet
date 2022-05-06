import torch
import torch.nn as nn
from .SCConv import SCBottleneck


class MSConv2DFusion(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = input_channels


        self.stride4x_group = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.stride8x_group = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.group_4x_block = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
        )

        self.group_8x_block = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
        )

        
        self.up_stride8_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.w_0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )

        self.w_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )



    def forward(self, batch_dict):
        stride8x_features = batch_dict['spatial_features']
        stride4x_features = batch_dict['spatial_4x_features']

        stride8x = self.stride8x_group(stride8x_features)
        stride4x = self.stride4x_group(stride4x_features)
        stride4x = self.group_4x_block(stride4x)

        stride8x = self.group_8x_block(stride8x)
        stride8x_4x = self.up_stride8_4(stride8x)
        weight4x = self.w_0(stride4x)
        weight8x = self.w_1(stride8x_4x)
        weights  = torch.softmax(torch.cat([weight4x, weight8x], dim=1), dim=1)
        output   = weights[:, 0:1, :, :] * stride4x + weights[:, 1:, :, :] * stride8x_4x
        batch_dict["spatial_features_2d"] = output.contiguous()
        return batch_dict