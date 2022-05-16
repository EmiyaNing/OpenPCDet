import torch
import torch.nn as nn
import torch.nn.functional as F

from .SCConv import SCBottleneck


class Self_SCONV(nn.Module):
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = 64
        self.groupv1_project = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.SELU(),
        )
        self.groupv1         = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
        )

        self.groupv1_combine = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )

        self.groupv2_project = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )
        self.groupv2         = nn.Sequential(
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
        )
        self.groupv2_upsample= nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )
        self.output_project  = nn.Sequential(
            nn.Conv2d(512, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.SELU(),
        )

        self.sub_net = nn.Sequential(
            nn.Conv2d(input_channels, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.SELU()
        )
        self.sub_head= nn.Sequential(
            nn.Conv2d(128, 3, 1, 1, padding=0),
        )

    def forward(self, data_dict):

        x     = data_dict["spatial_features"]
        sub_x = data_dict["spatial_features"]

        if self.training:
            sub_res_temp = self.sub_net(sub_x)
            sub_foreground_res = self.sub_head(sub_res_temp)

            data_dict['sub_foreground_segment_result'] = sub_foreground_res
        else:
            sub_res_temp = self.sub_net(sub_x)
            sub_for_resu = self.sub_head(sub_res_temp)
            sub_mask     = torch.mean(F.sigmoid(sub_for_resu), dim=1)
            sub_mask     = sub_mask.unsqueeze(1)
            x            = (1 + sub_mask) * x


        x     = self.groupv1_project(x)
        groupv1_temp = self.groupv1(x)
        groupv1_out  = self.groupv1_combine(groupv1_temp)


        groupv2_temp = self.groupv2_project(groupv1_temp)
        groupv2_temp = self.groupv2(groupv2_temp)
        groupv2_out  = self.groupv2_upsample(groupv2_temp)


        fusion_map   = torch.cat([groupv1_out, groupv2_out], dim = 1)
        output       = self.output_project(fusion_map)

        data_dict["spatial_features_2d"] = output.contiguous()


        return data_dict



class Self_SCONVV2(nn.Module):
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = 64
        self.groupv1_project = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.SELU(),
        )
        self.groupv1         = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
        )

        self.groupv1_combine = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )

        self.groupv2_project = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )
        self.groupv2         = nn.Sequential(
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
        )
        self.groupv2_upsample= nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )
        self.output_project  = nn.Sequential(
            nn.Conv2d(512, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.SELU(),
        )

        self.sub_head = nn.Sequential(
            nn.Conv2d(input_channels, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.Conv2d(128, 18, 1, 1, padding=0),
        )

    def forward(self, data_dict):

        x     = data_dict["spatial_features"]
        sub_x = data_dict["spatial_features"]

        if self.training:
            sub_res = self.sub_head(sub_x)
            data_dict['sub_foreground_segment_result'] = sub_res




        x     = self.groupv1_project(x)
        groupv1_temp = self.groupv1(x)
        groupv1_out  = self.groupv1_combine(groupv1_temp)


        groupv2_temp = self.groupv2_project(groupv1_temp)
        groupv2_temp = self.groupv2(groupv2_temp)
        groupv2_out  = self.groupv2_upsample(groupv2_temp)


        fusion_map   = torch.cat([groupv1_out, groupv2_out], dim = 1)
        output       = self.output_project(fusion_map)

        data_dict["spatial_features_2d"] = output.contiguous()


        return data_dict



class Self_SCONVV3(nn.Module):
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = 64
        self.groupv1_project = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.SELU(),
        )
        self.groupv1         = nn.Sequential(
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(128, 32, 1, norm_layer=nn.BatchNorm2d),
        )

        self.groupv1_combine = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )

        self.groupv2_project = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )
        self.groupv2         = nn.Sequential(
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
            SCBottleneck(256, 64, 1, norm_layer=nn.BatchNorm2d),
        )
        self.groupv2_upsample= nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )
        self.output_project  = nn.Sequential(
            nn.Conv2d(512, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.SELU(),
        )

        self.sub_head = nn.Sequential(
            nn.Conv2d(input_channels, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.Conv2d(128, 18, 1, 1, padding=0),
        )

        self.sub_box  = nn.Sequential(
            nn.Conv2d(input_channels, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.Conv2d(128, 42, 1, 1, padding=0),
        )

    def forward(self, data_dict):

        x     = data_dict["spatial_features"]
        sub_x = data_dict["spatial_features"]

        if self.training:
            sub_res = self.sub_head(sub_x)
            data_dict['sub_cls_result'] = sub_res
            sub_reg = self.sub_box(sub_x)
            data_dict['sub_box_reuslt'] = sub_reg




        x     = self.groupv1_project(x)
        groupv1_temp = self.groupv1(x)
        groupv1_out  = self.groupv1_combine(groupv1_temp)


        groupv2_temp = self.groupv2_project(groupv1_temp)
        groupv2_temp = self.groupv2(groupv2_temp)
        groupv2_out  = self.groupv2_upsample(groupv2_temp)


        fusion_map   = torch.cat([groupv1_out, groupv2_out], dim = 1)
        output       = self.output_project(fusion_map)

        data_dict["spatial_features_2d"] = output.contiguous()


        return data_dict