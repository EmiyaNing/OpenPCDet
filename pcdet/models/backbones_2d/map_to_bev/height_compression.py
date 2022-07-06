import torch
import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class HeightCompressionWithBEV(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.channel_compress = nn.Conv2d(self.num_bev_features * 2, self.num_bev_features, 1, 1)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                pred_bev_tensor: bev forward tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        bev_features          = batch_dict['pred_bev_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        spatial_features = torch.cat([spatial_features, bev_features], 1)
        spatial_features = self.channel_compress(spatial_features)

        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict



class HeightCompressionMultiScale(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.channel_compress_4x = nn.Conv2d(11 * 64, self.num_bev_features, 1, 1)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        downsample4x_tensor  = batch_dict['multi_scale_3d_features']['x_conv3'].dense()
        N4, C4, D4, H4, W4   = downsample4x_tensor.shape
        stride4x_features    = downsample4x_tensor.view(N4, C4 * D4, H4, W4)
        stride4x_features    = self.channel_compress_4x(stride4x_features)
        batch_dict['spatial_4x_features'] = stride4x_features
        return batch_dict




class HeightCompressionFPN(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.channel_compress_2x = nn.Conv2d(5 * 32, self.num_bev_features, 1, 1)
        self.channel_compress_4x = nn.Conv2d(5 * 32, self.num_bev_features, 1, 1)
        self.channel_compress_8x = nn.Conv2d(256, self.num_bev_features, 1, 1)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        spatial_features = self.channel_compress_8x(spatial_features)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        downsample4x_tensor  = batch_dict['multi_scale_3d_features']['stride4_down'].dense()
        downsample2x_tensor  = batch_dict['multi_scale_3d_features']['stride2_down'].dense()
        N4, C4, D4, H4, W4   = downsample4x_tensor.shape
        stride4x_features    = downsample4x_tensor.view(N4, C4 * D4, H4, W4)
        stride4x_features    = self.channel_compress_4x(stride4x_features)
        N2, C2, D2, H2, W2   = downsample2x_tensor.shape
        stride2x_features    = downsample2x_tensor.view(N2, C2 * D2, H2, W2)
        stride2x_features    = self.channel_compress_2x(stride2x_features)
        batch_dict['spatial_4x_features'] = stride4x_features
        batch_dict['spatial_2x_features'] = stride2x_features
        return batch_dict