from statistics import harmonic_mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils


class AnchorHeadHarmony(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_normized_weight(self):
        '''
            Get normized weight for regression tasks.
        '''
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        positives      = box_cls_labels > 0
        reg_weights    = positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights    /= torch.clamp(pos_normalizer, min=1.0)

        return reg_weights


    def get_task_constrastive_loss(self):
        cls_preds = F.sigmoid(self.forward_ret_dict['H_cls_preds'])
        box_preds = self.forward_ret_dict['H_box_preds']
        gt_boxes  = self.forward_ret_dict['gt_boxes'][:, :, :7]
        batch_size= self.forward_ret_dict['batch_size']

        reg_weights    = self.get_normized_weight()
        pos_pred_masks = reg_weights > 0
        task_constrast_loss    = []

        for bs_idx in range(batch_size):
            bs_pos_pred_masks = pos_pred_masks[bs_idx]
            bs_gt_boxes       = gt_boxes[bs_idx]
            bs_box_preds      = box_preds[bs_idx][bs_pos_pred_masks]
            bs_cls_preds      = cls_preds[bs_idx][bs_pos_pred_masks]
            bs_con_preds, _   = torch.max(bs_cls_preds, dim=-1)

            normalize_count   = bs_pos_pred_masks.sum()

            bs_iou_ground     = iou3d_nms_utils.boxes_iou3d_gpu(bs_box_preds, bs_gt_boxes)
            bs_iou_ground, _  = bs_iou_ground.max(dim=-1, keepdim=True)

            bs_gamma_e        = torch.exp(-torch.sum(bs_cls_preds * torch.log(bs_cls_preds), dim=-1))
            bs_constrast_elem = torch.clamp(torch.abs(bs_con_preds - bs_iou_ground) - 0.2, min=0)
            bs_constrast_loss = 1/(1 + bs_gamma_e) * bs_constrast_elem
            bs_constrast_loss = bs_constrast_loss.sum() / normalize_count
            task_constrast_loss.append(bs_constrast_loss)

        task_constractive_loss = sum(task_constrast_loss) / batch_size

        tb_dict ={
            'task_constractive_loss': task_constractive_loss.item(),
        }
        return task_constractive_loss, tb_dict

    def get_harmonic_iou_loss(self):
        box_preds = self.forward_ret_dict['H_box_preds']
        gt_boxes  = self.forward_ret_dict['gt_boxes'][:, :, :7]
        batch_size= self.forward_ret_dict['batch_size']

        reg_weights     = self.get_normized_weight()
        pos_pred_masks  = reg_weights > 0
        harmonic_iou_list = []

        for bs_idx in range(batch_size):
            bs_pos_pred_masks = pos_pred_masks[bs_idx]
            bs_box_preds      = box_preds[bs_idx][bs_pos_pred_masks]
            bs_gt_boxes       = gt_boxes[bs_idx]
            normalize_count   = bs_pos_pred_masks.sum()

            bs_iou_ground     = iou3d_nms_utils.boxes_iou3d_gpu(bs_box_preds, bs_gt_boxes)
            bs_iou_ground, _  = bs_iou_ground.max(dim=-1, keepdim=True)
            bs_harmonic_iou_loss = torch.pow(1 + bs_iou_ground, 0.8) * (1 - bs_iou_ground)
            bs_harmonic_iou_loss = bs_harmonic_iou_loss.sum() / normalize_count
            harmonic_iou_list.append(bs_harmonic_iou_loss)

        harmonic_iou_loss = sum(harmonic_iou_list) / batch_size

        tb_dict = {
            'harmonic_iou_loss': harmonic_iou_loss.item(),
        }
        return harmonic_iou_loss, tb_dict





    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        gamma_reg = torch.exp(-cls_loss)
        gamma_cls = torch.exp(-box_loss)
        rpn_loss  = (1 + gamma_reg) * cls_loss + (1 + gamma_cls) * box_loss
        tb_dict.update(tb_dict_box)
        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('task_constractive', None) is not None:
            weights_tasks = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['task_constractive']
            tasks_loss, tb_task_dict = self.get_task_constrastive_loss()
            rpn_loss += tasks_loss * weights_tasks
            tb_dict.update(tb_task_dict)

        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('harmonic_iou_loss', None) is not None:
            weights_iou   = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['harmonic_iou_loss']
            harmonic_iou_loss, tb_iou_dict = self.get_harmonic_iou_loss()
            rpn_loss += harmonic_iou_loss * weights_iou
            tb_dict.update(tb_iou_dict)

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['batch_size'] = data_dict['batch_size']
        self.forward_ret_dict['gt_boxes']   = data_dict['gt_boxes']
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            self.forward_ret_dict['H_cls_preds'] = batch_cls_preds
            self.forward_ret_dict['H_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
