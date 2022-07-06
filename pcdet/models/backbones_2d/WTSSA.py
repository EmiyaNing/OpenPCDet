import os
import copy
import torch
import torch.nn as nn
import numpy as np

from .network_blocks import BaseConv, CoordAtt
from .bev_transformer import DropPath
from .swin import BasicLayer

class CoorSWINNet(nn.Module):
    '''
        Coordinate_SSD
    '''
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        dim = input_channels
        out_dim    = dim
        self.project     = nn.Conv2d(256, 128, 1)
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