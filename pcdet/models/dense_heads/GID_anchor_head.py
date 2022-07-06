import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils 
from ...utils import odiou_loss

class GID_AnchorHeadSingle(AnchorHeadTemplate):
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
            nn.Conv2d(
                input_channels, 256, 1, 1
            ),
            nn.Conv2d(
                256, 256, 3, 1, 1
            ),
            nn.Conv2d(
                256, 256, 3, 1, 1
            )
        )

        self.reg_stems = nn.Sequential(
            nn.Conv2d(
                input_channels, 256, 1, 1
            ),
            nn.Conv2d(
                256, 256, 3, 1, 1
            ),
            nn.Conv2d(
                256, 256, 3, 1, 1
            )
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

        self.odiou_loss = odiou_loss.odiou_3D()
        self.knowledge_forward_rect = {}
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def nn_distance(self, student_box, teacher_box, iou_thres=0.7):
        iou_ground =  iou3d_nms_utils.boxes_iou3d_gpu(student_box, teacher_box)
        iou1, idx1 = torch.max(iou_ground, dim=1)
        iou2, idx2 = torch.max(iou_ground, dim=0)
        mask1, mask2 = iou1 > iou_thres, iou2 > iou_thres
        # filter box by iou_thresh....
        iou_ground = iou_ground[mask1]
        iou_ground = iou_ground[:, mask2]
        if iou_ground.shape[0] == 0 or iou_ground.shape[1] == 0:  # for unlabeled data (some scenes wo cars)
            return [None] * 5

        iou1, idx1 = torch.max(iou_ground, dim=1)
        iou2, idx2 = torch.max(iou_ground, dim=0)
        val_box1, val_box2 = student_box[mask1], teacher_box[mask2]
        aligned_box1, aligned_box2 = val_box1[idx2], val_box2[idx1]
        box1, box2 = self.add_sin_difference(val_box1, aligned_box2)
        box_cosistency_loss = self.reg_loss_func(box1, box2)
        box_cosistency_loss = box_cosistency_loss.sum() / box_cosistency_loss.shape[0]
        return box_cosistency_loss, idx1, idx2, mask1, mask2



    def consistency_loss(self):
        # First get the decoded predicts of box and cls.
        cost_function = torch.nn.SmoothL1Loss()
        student_cls = self.forward_ret_dict['batch_cls_preds']
        student_box = self.forward_ret_dict['batch_box_preds']
        student_dir = self.forward_ret_dict['batch_dir_cls_preds']
        teacher_cls = self.knowledge_forward_rect['batch_cls_preds']
        teacher_box = self.knowledge_forward_rect['batch_box_preds']
        teacher_dir = self.knowledge_forward_rect['batch_dir_cls_preds']
        batch_sz, height, width, _ = student_dir.shape
        student_dir = student_dir.view(batch_sz, height * width * self.num_class * 2, -1)
        teacher_dir = teacher_dir.view(batch_sz, height * width * self.num_class * 2, -1)
        # Second Get the max cls score of each class in student_cls and teacher_cls
        student_score = torch.max(student_cls, dim=-1)[0]
        teacher_score = torch.max(teacher_cls, dim=-1)[0]
        batch_sz, _, _ = student_cls.shape
        batch_box_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_cls_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_dir_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        # Third Get the student_cls mask and teacher_cls mask
        for i in range(batch_sz):
            student_mask = student_score[i] > 0.3
            teacher_mask = teacher_score[i] > 0.3
            if student_mask.sum() > 0 and teacher_mask.sum() > 0:

                student_cls_filter = student_cls[i][student_mask]
                student_box_filter = student_box[i][student_mask]
                student_dir_filter = student_dir[i][student_mask]
                teacher_cls_filter = teacher_cls[i][teacher_mask]
                teacher_box_filter = teacher_box[i][teacher_mask]
                teacher_dir_filter = teacher_dir[i][teacher_mask]
                # See how many box will be remain....
                con_box_loss, idx1, idx2, mask1, mask2 = self.nn_distance(student_box_filter, teacher_box_filter)
                if con_box_loss is None:
                    continue
                student_cls_selected = torch.sigmoid(student_cls_filter[mask1])
                teacher_cls_selected = torch.sigmoid(teacher_cls_filter[mask2][idx1])
                student_dir_selected = F.softmax(student_dir_filter[mask1], dim=-1)
                teacher_dir_selected = F.softmax(teacher_dir_filter[mask2][idx1], dim=-1)
                batch_box_loss += con_box_loss
                batch_cls_loss += cost_function(student_cls_selected, teacher_cls_selected)
                batch_dir_loss += cost_function(student_dir_selected, teacher_dir_selected)
        
        consistency_loss = (batch_dir_loss + batch_box_loss + batch_cls_loss) / batch_sz
        tb_dict = {
            'consistency_loss': consistency_loss.item(),
            'consistency_dir_loss': batch_dir_loss.item(),
            'consistency_box_loss': batch_box_loss.item(),
            'consistency_cls_loss': batch_cls_loss.item(),
        }
        return consistency_loss, tb_dict


    def get_iou_loss(self):
        cost_function = torch.nn.SmoothL1Loss()
        iou_preds      = self.forward_ret_dict['iou_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        box_preds      = self.forward_ret_dict['batch_box_preds']
        box_targets    = self.forward_ret_dict['box_reg_targets']
        batch_size, _, _, _ = iou_preds.shape
        positives = box_cls_labels > 0
        reg_weights = positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        pos_pred_mask = reg_weights > 0
        iou_pos_preds = iou_preds.view(batch_size, -1, 1)[pos_pred_mask]
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
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        box_preds      = self.forward_ret_dict['batch_box_preds']

        nan_indictor   = torch.isnan(box_preds)
        nan_indictor   = nan_indictor.float()
        nan_flag       = nan_indictor.sum()
        box_targets    = self.forward_ret_dict['box_reg_targets']
        odiou_loss     = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_size, _, _ = box_preds.shape
        positives = box_cls_labels > 0
        reg_weights = positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        pos_pred_mask = reg_weights > 0
        if pos_pred_mask.sum() > 0:
            qboxes = box_preds[pos_pred_mask]
            gboxes = box_targets[pos_pred_mask]
            weight = reg_weights[pos_pred_mask]
            odiou_loss += self.odiou_loss(qboxes, gboxes, weight.unsqueeze(dim=-1), batch_size)
        tb_dicts = {
            'odiou_loss': odiou_loss.item()
        }
        return odiou_loss, tb_dicts


    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        if not self.model_cfg.TEACHER:
            con_weight = self.knowledge_forward_rect['consistency_weight']
            kd_loss, tb_con_dicts  = self.consistency_loss()
            rpn_loss = cls_loss + box_loss + kd_loss * con_weight
            tb_dict.update(tb_con_dicts)
            if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight'] is not None:
                iou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
                iou_loss, tb_iou_dicts = self.get_iou_loss()
                rpn_loss += iou_loss * iou_weight
                tb_dict.update(tb_iou_dicts)
        else:
            rpn_loss = cls_loss + box_loss
            if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight'] is not None:
                iou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
                iou_loss, tb_iou_dicts = self.get_iou_loss()
                rpn_loss += iou_loss * iou_weight
                tb_dict.update(tb_iou_dicts)
            if self.model_cfg.get('oiou_weight', None) is not None:
                odiou_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['oiou_weight']
                odiou_loss, tb_odiou_dicts = self.get_odiou_loss()
                odiou_loss = odiou_loss * odiou_weight
                rpn_loss += odiou_loss[0]
                tb_dict.update(tb_odiou_dicts)
            

        tb_dict.update(tb_dict_box)

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict



    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_temp  = self.cls_stems(spatial_features_2d)
        reg_temp  = self.reg_stems(spatial_features_2d)

        cls_preds = self.conv_cls(cls_temp)
        iou_preds = self.conv_iou(cls_temp)
        box_preds = self.conv_box(reg_temp)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()
        batch_size, _, _, _ = iou_preds.shape

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['iou_preds'] =  iou_preds
        self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
        if self.training:
            teacher_flag = data_dict['is_ema']

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(reg_temp)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

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
                if teacher_flag:
                    self.knowledge_forward_rect['batch_cls_preds'] = data_dict['ema_cls_preds']
                    self.knowledge_forward_rect['batch_box_preds'] = data_dict['ema_box_preds']
                    self.knowledge_forward_rect['batch_dir_cls_preds'] = data_dict['ema_dir_cls_preds']
                    self.knowledge_forward_rect['consistency_weight'] = data_dict['consistency_weight']

            data_dict['cls_preds_normalized'] = False

        return data_dict

