import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_blocks import CoordAtt

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=True,
                 norm_layer=None):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.group_width = group_width
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=False),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)


        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)


        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_attention_map(tensor_map):
    tensor_map = torch.pow(torch.abs(tensor_map),2)
    attention  = torch.mean(tensor_map, dim=1)
    return attention

class SCConv2D(nn.Module):
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

    def forward(self, data_dict):

        x = data_dict["spatial_features"]
        x = self.groupv1_project(x)
        groupv1_temp = self.groupv1(x)
        groupv1_out  = self.groupv1_combine(groupv1_temp)
        attention_group1 = get_attention_map(groupv1_out)

        groupv2_temp = self.groupv2_project(groupv1_temp)
        groupv2_temp = self.groupv2(groupv2_temp)
        groupv2_out  = self.groupv2_upsample(groupv2_temp)
        attention_group2 = get_attention_map(groupv2_out)

        fusion_map   = torch.cat([groupv1_out, groupv2_out], dim = 1)
        output       = self.output_project(fusion_map)
        attention_group3 = get_attention_map(output)
        data_dict["spatial_features_2d"] = output.contiguous()
        data_dict["attention_group1"] = attention_group1
        data_dict["attention_group2"] = attention_group2
        data_dict["attention_group3"] = attention_group3

        return data_dict


class SCConv2DV2(nn.Module):
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
        )

        self.spatial_block   = CoordAtt(128, 128, 16)

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

    def forward(self, data_dict):

        x = data_dict["spatial_features"]
        x = self.groupv1_project(x)
        groupv1_temp = self.groupv1(x)
        spatial_mask = self.spatial_block(groupv1_temp)
        groupv1_temp = spatial_mask * groupv1_temp
        groupv1_out  = self.groupv1_combine(groupv1_temp)

        groupv2_temp = self.groupv2_project(groupv1_temp)
        groupv2_temp = self.groupv2(groupv2_temp)
        groupv2_out  = self.groupv2_upsample(groupv2_temp)

        fusion_map   = torch.cat([groupv1_out, groupv2_out], dim = 1)
        output       = self.output_project(fusion_map)
        data_dict["spatial_features_2d"] = output.contiguous()

        return data_dict

class SCConv2DV3(nn.Module):
    def __init__(self,  model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = 256
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
            nn.Conv2d(512, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.SELU(),
        )

    def forward(self, data_dict):

        x = data_dict["spatial_features"]
        x = self.groupv1_project(x)
        groupv1_temp = self.groupv1(x)
        groupv1_out  = self.groupv1_combine(groupv1_temp)

        groupv2_temp = self.groupv2_project(groupv1_temp)
        groupv2_temp = self.groupv2(groupv2_temp)
        groupv2_out  = self.groupv2_upsample(groupv2_temp)

        fusion_map   = torch.cat([groupv1_out, groupv2_out], dim = 1)
        output       = self.output_project(fusion_map)
        data_dict["spatial_features_2d"] = output.contiguous()

        return data_dict