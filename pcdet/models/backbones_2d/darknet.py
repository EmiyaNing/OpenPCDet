import torch
from torch import nn

from .network_blocks import BaseConv, DWConv, Focus, SPPBottleneck, CoordAtt
from .bev_transformer import TransBEVBackbone

class CSPDarknet(nn.Module):
    def __init__(
        self,
        model_cfg
    ):
        self.model_cfg = model_cfg
        dep_mul = model_cfg.DEPTH
        wid_mul = model_cfg.WIDTH
        super().__init__()
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES * 2
        depthwise = model_cfg.DEPTHWISE
        act  = model_cfg.CONV_ACTION
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels*2, ksize=3, act=act)
        self.dark2= SPPBottleneck(base_channels*2, base_channels*4, activation=act)

        self.compress     = nn.Conv2d(512, self.model_cfg.NUM_BEV_FEATURES, 1, 1)
        # transformer module
        self.rpn_backbone = TransBEVBackbone(model_cfg, self.model_cfg.NUM_BEV_FEATURES)




    def forward(self, data_dict):
        x = data_dict['bev']
        x = self.stem(x)
        x = self.dark2(x)
        x = x.permute(0, 1, 3, 2)
        features = data_dict['spatial_features']
        x = torch.cat([features, x], 1)
        x = self.compress(x)
        data_dict["spatial_features"] = x
        data_dict = self.rpn_backbone(data_dict)

        return data_dict


class CoordTransformer(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        super().__init__()
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES * 2
        act  = model_cfg.CONV_ACTION

        self.stem = Focus(3, 256, ksize=3, act=act)
        self.compress     = nn.Conv2d(512, self.model_cfg.NUM_BEV_FEATURES, 1, 1)
        # transformer module
        self.rpn_backbone = TransBEVBackbone(model_cfg, self.model_cfg.NUM_BEV_FEATURES)
        self.coordinate   = CoordAtt(self.model_cfg.NUM_BEV_FEATURES, self.model_cfg.NUM_BEV_FEATURES * 2)

    def forward(self, data_dict):
        x = data_dict['bev']
        x = self.stem(x)
        x = x.permute(0, 1, 3, 2)
        features = data_dict['spatial_features']
        mask     = self.coordinate(features)
        x = torch.cat([features, x], 1)
        x = self.compress(x)
        data_dict["spatial_features"] = x
        data_dict = self.rpn_backbone(data_dict)
        result_sp = data_dict["spatial_features"]
        result_sp = result_sp * mask
        data_dict["spatial_features"] = result_sp
        return data_dict
