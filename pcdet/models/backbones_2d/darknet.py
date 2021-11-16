from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

class CSPDarknet(nn.Module):
    def __init__(
        self,
        model_cfg,
    ):
        self.model_cfg = model_cfg
        dep_mul = model_cfg.DEPTH
        wid_mul = model_cfg.WIDTH
        super().__init__()
        self.out_features = model_cfg.OUT_FEATURES
        depthwise = model_cfg.DEPTHWISE
        act  = model_cfg.ACTION
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
            SPPBottleneck(base_channels * 4, base_channels * 4, activation=act),
        )


    def forward(self, data_dict):
        x = data_dict['bev']
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        data_dict["pred_bev_tensor"] = x

        return data_dict