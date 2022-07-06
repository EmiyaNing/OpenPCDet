import torch
import torch.nn as nn
from .SCConv import SCBottleneck


class MultiScale_SCConv2D(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = 256


        self.stride4x_group = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.stride8x_group = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, 1),
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



    def forward(self, batch_dict):
        stride8x_features = batch_dict['spatial_features']
        stride4x_features = batch_dict['spatial_4x_features']

        stride8x = self.stride8x_group(stride8x_features)
        stride4x = self.stride4x_group(stride4x_features)
        stride4x = self.group_4x_block(stride4x)

        stride8x = self.group_8x_block(stride8x)
        stride8x_4x = self.up_stride8_4(stride8x)
        output   = torch.cat([stride4x, stride8x_4x], dim=1)


        batch_dict["spatial_features_2d"] = output.contiguous()
        return batch_dict


class MultiAttentionFusion(nn.Module):
    '''
        Lightweigh MultiSCONVNet, should use with HCMultiScalev2.....
    '''
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
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.group_4x_block = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            nn.Conv2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.group_8x_block = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
        )

        
        self.up_stride8_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.w_0 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )

        self.w_1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
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
        x_weight_0 = self.w_0(stride4x)
        x_weight_1 = self.w_1(stride8x_4x)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        output = stride4x * x_weight[:, 0:1, :, :] + stride8x_4x * x_weight[:, 1:, :, :]
        batch_dict["spatial_features_2d"] = output.contiguous()
        return batch_dict



class MultiSCONVv2(nn.Module):
    '''
        Lightweigh MultiSCONVNet, should use with HCMultiScalev2.....
    '''
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = 256


        self.stride4x_group = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.stride8x_group = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.group_4x_block = nn.Sequential(
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



    def forward(self, batch_dict):
        stride8x_features = batch_dict['spatial_features']
        stride4x_features = batch_dict['spatial_4x_features']

        stride8x = self.stride8x_group(stride8x_features)
        stride4x = self.stride4x_group(stride4x_features)

        #stride4x = self.group_4x_block(stride4x)

        stride8x = self.group_8x_block(stride8x)
        stride8x_4x = self.up_stride8_4(stride8x)
        output   = torch.cat([stride4x, stride8x_4x], dim=1)


        batch_dict["spatial_features_2d"] = output.contiguous()
        return batch_dict


class MultiSCConvFPN(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = input_channels

        self.stride2x_group = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.stride4x_group = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.stride8x_group = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


        self.group_2x_block = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.group_4x_block = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
        )

        self.group_8x_block = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.w_stride2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )

        self.w_stride4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )

        self.w_stride8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            nn.BatchNorm2d(1),
        )
        



    def forward(self, batch_dict):
        stride8x_features = batch_dict['spatial_features']
        stride4x_features = batch_dict['spatial_4x_features']
        stride2x_features = batch_dict['spatial_2x_features']
        stride2x  = self.stride2x_group(stride2x_features)
        stride4x  = self.stride4x_group(stride4x_features)
        stride8x  = self.stride8x_group(stride8x_features)
        temp_stride2x = self.group_2x_block(stride2x)
        temp_stride4x = self.group_4x_block(stride4x)
        temp_stride8x = self.group_8x_block(stride8x)
        weight2x  = self.w_stride2(temp_stride2x)
        weight4x  = self.w_stride4(temp_stride4x)
        weight8x  = self.w_stride8(temp_stride8x)
        weight    = torch.sigmoid(torch.cat([weight2x, weight4x, weight8x], dim=1))
        output    = temp_stride2x * weight[:, 0:1, :, :] + temp_stride4x * weight[:, 1:2, :, :] + temp_stride8x * weight[:, 2:, :, :]
        batch_dict["spatial_features_2d"] = output.contiguous()
        return batch_dict