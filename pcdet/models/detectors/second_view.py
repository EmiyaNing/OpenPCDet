import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.lib.function_base import disp
from .detector3d_template import Detector3DTemplate
from ...utils import common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ..model_utils.meter_utils import AverageMeter


class SECONDNet_view(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.voxel_size  = [0.05, 0.05, 0.1]
        self.point_cloud_range = [0, -40, -3, 70.4, 40, 1]
        self.cost_function = nn.CrossEntropyLoss(reduction='sum')
        self.forward_ret_dict = {}
        self.center_cost_function = nn.SmoothL1Loss(reduction='none')

        sigma = torch.randn(2, requires_grad=True)
        self.sigma = nn.Parameter(sigma)
        self.rpn_total_loss= AverageMeter()
        self.stride3x_loss = AverageMeter()
        self.stride4x_loss = AverageMeter()

    def get_single_stride_structure_awareness_loss(self, coords_tensor, voxel_center_tensor, segment_preds, center_preds):
        batch_size = self.forward_ret_dict['batch_size']
        single_stride_loss = 0
        for k in range(batch_size):
            batch_one_mask = coords_tensor[:, 0] == k
            voxel_cls_label= batch_one_mask.new_zeros(batch_one_mask.sum()).long()
            one_masks      = voxel_cls_label.new_ones(batch_one_mask.sum()).long()
            center_label   = batch_one_mask.new_zeros(batch_one_mask.sum(), 3).float()
            segment_result = F.sigmoid(segment_preds[batch_one_mask])

            batch_one_voxel_xyz = voxel_center_tensor[batch_one_mask]
            center_result       = center_preds[batch_one_mask] + batch_one_voxel_xyz


            batch_one_gt_boxes  = self.forward_ret_dict['gt_boxes'][k:k+1]
            box_idxs_of_pts     = roiaware_pool3d_utils.points_in_boxes_gpu(batch_one_voxel_xyz.unsqueeze(0), batch_one_gt_boxes[:, :, :7].contiguous()).long().squeeze(dim=0)
            fg_flag             = (box_idxs_of_pts >= 0)

            gt_box_of_fg_voxels = batch_one_gt_boxes[0][box_idxs_of_pts[fg_flag]]
            

            voxel_cls_label[fg_flag] = one_masks[fg_flag]
            center_label[fg_flag]    = gt_box_of_fg_voxels[:, :3].float()

            positives           = (voxel_cls_label > 0)
            negative_cls_weights= (voxel_cls_label == 0) * 1.0
            cls_weights         = (1.0 * negative_cls_weights + 1.0 * positives).float()
            pos_normalizer      = positives.sum(dim=0).float()
            cls_weights         /= torch.clamp(pos_normalizer, min=1.0)

            reg_weights         = positives.float()
            reg_weights         /= pos_normalizer

            one_hot_targets     = voxel_cls_label.new_zeros(*list(voxel_cls_label.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (voxel_cls_label * (voxel_cls_label >= 0).long()).unsqueeze(-1).long(), 1.0)
            one_hot_targets     = one_hot_targets[:, 1:]
            single_batch_loss   = self.dense_head.cls_loss_func(segment_result.unsqueeze(0), one_hot_targets.unsqueeze(0), weights=cls_weights.unsqueeze(0)).mean(dim=-1).sum()
            single_batch_center_loss = (self.center_cost_function(center_result, center_label) * reg_weights.unsqueeze(-1)).sum()

            single_stride_loss  += single_batch_loss * 0.9 / batch_size
            single_stride_loss  += single_batch_center_loss * 2 / batch_size
        return single_stride_loss


    def get_structure_awareness_loss(self):
        sparse_voxel4 = self.forward_ret_dict['x_conv4']
        sparse_voxel3 = self.forward_ret_dict['x_conv3']
        segment_preds3= self.forward_ret_dict['segment_stride3']
        segment_preds4= self.forward_ret_dict['segment_stride4']
        center_preds3 = self.forward_ret_dict['center_stride3']
        center_preds4 = self.forward_ret_dict['center_stride4']

        coords_stride4  = sparse_voxel4.indices
        voxel_center4   = common_utils.get_voxel_centers(
            coords_stride4[:, 1:4],
            downsample_times=8,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )

        coords_stride3  = sparse_voxel3.indices
        voxel_center3   = common_utils.get_voxel_centers(
            coords_stride3[:, 1:4],
            downsample_times=4,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )

        structure_loss3  = self.get_single_stride_structure_awareness_loss(coords_stride3, voxel_center3, segment_preds3, center_preds3)
        structure_loss4  = self.get_single_stride_structure_awareness_loss(coords_stride4, voxel_center4, segment_preds4, center_preds4)

        total_loss       = structure_loss3 + structure_loss4
        self.stride3x_loss.update(structure_loss3.item())
        self.stride4x_loss.update(structure_loss4.item())
        tb_dict = {
            'total_loss': total_loss,
            'stride3x_loss': self.stride3x_loss.avg,
            'stride4x_loss': self.stride4x_loss.avg,
        }
        return total_loss, tb_dict
    


    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        self.forward_ret_dict['x_conv4'] = batch_dict['multi_scale_3d_features']['x_conv4']
        self.forward_ret_dict['x_conv3'] = batch_dict['multi_scale_3d_features']['x_conv3']
        self.forward_ret_dict['segment_stride3'] = batch_dict['segment_preds']['segment_stride3']
        self.forward_ret_dict['segment_stride4'] = batch_dict['segment_preds']['segment_stride4']
        self.forward_ret_dict['center_stride3']  = batch_dict['center_preds']['center_stride3']
        self.forward_ret_dict['center_stride4']  = batch_dict['center_preds']['center_stride4']
        self.forward_ret_dict['batch_size'] = batch_dict['batch_size']
        self.forward_ret_dict['gt_boxes']   = batch_dict['gt_boxes']
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_seg, tb_seg_dict = self.get_structure_awareness_loss()
        self.rpn_total_loss.update(loss_rpn.item())

        tb_dict = {
            'loss_rpn': self.rpn_total_loss.avg,
            **tb_dict
        }
        tb_dict.update(tb_seg_dict)

        loss = loss_rpn + loss_seg 
        return loss, tb_dict, disp_dict
