import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils 

class ODiou_Head(AnchorHeadTemplate):
    '''
        This class will implement the distillation loss.
        This class do not support multihead.....
    '''
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.cls_stems = nn.Sequential(
            nn.Conv2d(input_channels, 256, 1, 1),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.Conv2d(256, 256, 3, 1, padding=1),
        )

        self.reg_stems = nn.Sequential(
            nn.Conv2d(input_channels, 256, 1, 1),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.Conv2d(256, 256, 3, 1, padding=1),
        )

        self.conv_cls = nn.Conv2d(
            256, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )

        self.conv_iou = nn.Conv2d(
            256, self.num_anchors_per_location,
            kernel_size=1
        )

        self.conv_box = nn.Conv2d(
            256, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_dir_cls = nn.Conv2d(
            256,
            self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
            kernel_size=1
        )

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)


    def get_normized_weight(self):
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        positives = box_cls_labels > 0
        reg_weights = positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        return reg_weights


    def get_iou_loss(self):
        cost_function = torch.nn.SmoothL1Loss()
        iou_preds      = self.forward_ret_dict['iou_preds']

        box_preds      = self.forward_ret_dict['batch_box_preds']
        box_targets    = self.forward_ret_dict['box_reg_targets']

        reg_weights   = self.get_normized_weight()
        pos_pred_mask = reg_weights > 0
        iou_pos_preds = iou_preds[pos_pred_mask]
        qboxes = box_preds[pos_pred_mask]
        gboxes = box_targets[pos_pred_mask]
        iou_ground =  iou3d_nms_utils.boxes_iou3d_gpu(qboxes, gboxes).detach()
        iou_ground, _ = iou_ground.max(dim=-1, keepdim=True)
        iou_pos_targets = 2 * iou_ground - 1
        iou_loss = cost_function(iou_pos_preds, iou_pos_targets) 
        tb_dicts = {
            'iou_loss': iou_loss.item()
        }
        return iou_loss, tb_dicts


    def get_odiou_loss(self):
        box_preds      = self.forward_ret_dict['batch_box_preds']
        box_targets    = self.forward_ret_dict['box_reg_targets']
        odiou_loss     = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_size, _, _ = box_preds.shape
        reg_weights    = self.get_normized_weight()
        pos_pred_mask  = reg_weights > 0
        tb_dicts = {}
        if pos_pred_mask.sum() > 0:
            preds_boxes = box_preds[pos_pred_mask]
            label_boxes = box_targets[pos_pred_mask]
            iou_ground  =  iou3d_nms_utils.boxes_iou3d_gpu(preds_boxes, label_boxes)
            iou_ground,_ = iou_ground.max(dim=-1, keepdim=False)
            weights     = reg_weights[pos_pred_mask]
            iou_loss    = ((1 - iou_ground) * weights).sum() / batch_size

            angle_diff  = preds_boxes[:, -1] - label_boxes[:, -1]
            angle_loss  = (1.25 * (1.0 - torch.abs(torch.cos(angle_diff)))* weights).sum() / batch_size

            odiou_loss  += iou_loss + angle_loss
        
            tb_dicts       = {
                'odiou_loss': odiou_loss.item(),
                'iou_loss': iou_loss.item(),
                'angle_loss': angle_loss.item(),
            }
            return odiou_loss, tb_dicts

        else:
            return odiou_loss, tb_dicts



    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        rpn_loss = box_loss + cls_loss
        tb_dict.update(tb_dict_box)


        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight'] is not None:
            iou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
            iou_loss, tb_iou_dicts = self.get_iou_loss()
            rpn_loss += iou_loss * iou_weight
            tb_dict.update(tb_iou_dicts)

        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['oiou_weight'] is not None:
            odiou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['oiou_weight']
            odiou_loss, tb_odiou_dicts = self.get_odiou_loss()
            odiou_loss = odiou_loss * odiou_weight
            rpn_loss += odiou_loss[0]
            tb_dict.update(tb_odiou_dicts)
            
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict



    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_temp  = self.cls_stems(spatial_features_2d)
        reg_temp  = self.reg_stems(spatial_features_2d)

        cls_preds = self.conv_cls(cls_temp)
        iou_preds = self.conv_iou(cls_temp)
        box_preds = self.conv_box(reg_temp)
        dir_cls_preds = self.conv_dir_cls(reg_temp)
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()
        batch_size, _, _, _ = iou_preds.shape

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['iou_preds'] =  iou_preds.view(batch_size, -1, 1)
        self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
        self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        if self.training:
            teacher_flag = data_dict['is_ema']


        if self.training:
            ### In here we should add some code to assign the target for soft label...
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
            data_dict['batch_dir_preds'] = dir_cls_preds
            data_dict['batch_box_ious']  = iou_preds.view(batch_size, -1, 1)
            if self.training:
                self.forward_ret_dict['batch_cls_preds'] = batch_cls_preds
                self.forward_ret_dict['batch_box_preds'] = batch_box_preds
                self.forward_ret_dict['batch_dir_cls_preds'] = dir_cls_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

class Iou_Head(AnchorHeadTemplate):
    '''
        This class will implement the distillation loss.
        This class do not support multihead.....
    '''
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

        self.conv_iou = nn.Conv2d(
            input_channels, self.num_anchors_per_location,
            kernel_size=1
        )

        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_dir_cls = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
            kernel_size=1
        )

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']


        cls_preds = self.conv_cls(spatial_features_2d)
        iou_preds = self.conv_iou(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()
        batch_size, _, _, _ = iou_preds.shape

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['iou_preds'] =  iou_preds.view(batch_size, -1, 1)
        if self.training:
            teacher_flag = data_dict['is_ema']


        if self.training:
            ### In here we should add some code to assign the target for soft label...
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            data_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_box_ious']  = iou_preds.view(batch_size, -1, 1)
            data_dict['cls_preds_normalized'] = False

        return data_dict