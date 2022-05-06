import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.lib.function_base import disp
from .detector3d_template import Detector3DTemplate
from ...utils import common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ..backbones_3d.pfe.residual_v2p_decoder import ResidualVoxelToPointDecoder


class FromVoxel2Point(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.forward_ret_dict = {}

        sigma = torch.randn(2, requires_grad=True)
        self.sigma = nn.Parameter(sigma)


    def forward(self, batch_dict):
        if self.training:
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)
            batch_dict = self.post_pfe(batch_dict)
            batch_dict = self.point_head(batch_dict)
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict   = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn / (2 * self.sigma[0] ** 2) + loss_point / (2 * self.sigma[1] ** 2) \
               + torch.log(1 + self.sigma[0]**2) + torch.log(1 + self.sigma[1]**2)
        return loss, tb_dict, disp_dict
