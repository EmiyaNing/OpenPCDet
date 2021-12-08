import torch
import torch.nn as nn
from numpy.lib.function_base import disp
from .detector3d_template import Detector3DTemplate


class PVSECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        sigma = torch.randn(2, requires_grad=True)
        self.sigma = nn.Parameter(sigma)

    def forward(self, batch_dict):
        if self.training:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            # Not forward the voxelsetabstraction and point head
            for cur_module in self.module_list:
                if cur_module.__class__.__name__ not in ['ResidualVoxelToPointDecoder','VoxelSetAbstraction','PointHeadSimple', 'PointHeadBox']:
                    batch_dict = cur_module(batch_dict)

            pred_bev_feature = batch_dict['spatial_features']
            pred_bev_result  = batch_dict['spatial_features_2d']
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            pred_dicts[0]['bev_feature'] = pred_bev_feature
            pred_dicts[0]['bev_feature_2d'] = pred_bev_result
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_point': loss_point.item(),
            **tb_dict
        }

        loss = loss_rpn / (2 * self.sigma[0] ** 2) + loss_point / (2 * self.sigma[1] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss, tb_dict, disp_dict
