import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_head_template import AnchorHeadTemplate
from ...ops.iou3d_nms import iou3d_nms_utils 
from ..model_utils import meter_utils


class Diversity_Head(AnchorHeadTemplate):
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
        self.conv_box = nn.Conv2d(
            256, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_dir_cls = nn.Conv2d(
            256,
            self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
            kernel_size=1
        )
        post_center_range = [0, -40.0, -5.0, 70.4, 40.0, 5.0]
        self.post_center_range = torch.tensor(post_center_range, dtype=torch.float).cuda()
        self.knowledge_forward_rect = {}
        self.init_weights()

        #self.kd_cls_meter   = meter_utils.AverageMeter()
        #self.kd_fea_meter   = meter_utils.AverageMeter()
        #self.kd_cls_fea_meter = meter_utils.AverageMeter()
        #self.kd_reg_fea_meter = meter_utils.AverageMeter()
        #self.kd_fea_total_meter = meter_utils.AverageMeter()
        #self.kd_con_meter   = meter_utils.AverageMeter()
        



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

    def get_normized_weight(self):
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        positives = box_cls_labels > 0
        reg_weights = positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        return reg_weights



    def consistency_loss(self):
        # First get the decoded predicts of box and cls.
        cost_function = torch.nn.SmoothL1Loss()
        reg_weight  = self.get_normized_weight()
        student_cls = self.forward_ret_dict['batch_cls_preds']
        student_box = self.forward_ret_dict['batch_box_preds']
        student_dir = self.forward_ret_dict['batch_dir_cls_preds']
        teacher_cls = self.knowledge_forward_rect['teacher_cls_pred']
        teacher_box = self.knowledge_forward_rect['teacher_box_pred']
        teacher_dir = self.knowledge_forward_rect['teacher_dir_pred']

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
            mask_stu = (student_box[i][:, :3] >= self.post_center_range[:3]).all(1)
            mask_stu &= (student_box[i][:, :3] <= self.post_center_range[3:]).all(1)
            student_mask &= mask_stu
            mask_tea = (teacher_box[i][:, :3] >= self.post_center_range[:3]).all(1)
            mask_tea &= (teacher_box[i][:, :3] <= self.post_center_range[3:]).all(1)
            teacher_mask &= mask_tea
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

        
        consistency_loss = (batch_dir_loss + batch_box_loss + batch_cls_loss ) / batch_sz

        tb_dict = {
            'consistency_loss': consistency_loss.item(),
            'consistency_dir_loss': batch_dir_loss.item(),
            'consistency_box_loss': batch_box_loss.item(),
            'consistency_cls_loss': batch_cls_loss.item(),
        }
        return consistency_loss, tb_dict




    def get_kd_reg_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.knowledge_forward_rect['box_reg_targets']
        box_cls_labels  = self.knowledge_forward_rect['box_cls_labels']

        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        anchors = torch.cat(self.anchors, dim=-3)

        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        teach_loss = loc_loss
        tb_dict = {
            'rpn_kd_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            teach_loss += dir_loss
            tb_dict['rpn_kd_loss_dir'] = dir_loss.item()

        return teach_loss, tb_dict


    def get_kd_cls_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.knowledge_forward_rect['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_kd_loss_cls': cls_loss.item(),
        }
        return cls_loss, tb_dict



    def get_hint_loss(self):
        '''
            Reference the NeuroIPS 2020 Richness Feature Knowledge Distillation...
            Using the diversity of cls pred to get the richness feature, we called them Diversity Feature.....
        '''
        cost_function   = nn.MSELoss()
        #cost_function   = nn.KLDivLoss(reduction='batchmean')
        student_feature = self.forward_ret_dict['student_feature']
        student_mask    = F.sigmoid(self.forward_ret_dict['cls_preds'])
        teacher_feature = self.knowledge_forward_rect['teacher_feature']
        teacher_mask    = F.sigmoid(self.knowledge_forward_rect['teacher_cls_pred'])
        teacher_mask = teacher_mask.view(teacher_mask.shape[0], self.num_anchors_per_location, 200, 176, teacher_mask.shape[2])
        student_mask = student_mask.view(student_mask.shape[0], 200, 176, self.num_anchors_per_location, -1)


        mask_filter_teacher, _ = torch.max(teacher_mask, dim=1, keepdim=True)
        mask_filter_teacher, _ = torch.max(mask_filter_teacher, dim=-1)
        mask_filter_student, _ = torch.max(student_mask, dim=-2, keepdim=True)
        mask_filter_student = mask_filter_student.permute(0, 3, 1, 2, 4)
        mask_filter_student, _ = torch.max(mask_filter_student, dim=-1)
        mask_filter = torch.abs(mask_filter_teacher - mask_filter_student)
        teacher_div_feature = mask_filter * teacher_feature
        student_div_feature = mask_filter * student_feature

        student_cls_temp = self.forward_ret_dict['student_cls_temp'] * mask_filter
        student_reg_temp = self.forward_ret_dict['student_reg_temp'] * mask_filter
        teacher_cls_temp = self.knowledge_forward_rect['teacher_head_cls_temp'] * mask_filter
        teacher_reg_temp = self.knowledge_forward_rect['teacher_head_reg_temp'] * mask_filter

        fea_loss = cost_function(student_div_feature, teacher_div_feature)
        
        cls_fea_loss = cost_function(student_cls_temp, teacher_cls_temp)
        reg_fea_loss = cost_function(student_reg_temp, teacher_reg_temp)

        fea_weight     = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kd_fea_weight']
        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('kd_cls_weight', None) is not None:
            cls_fea_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kd_cls_weight']
        else:
            cls_fea_weight = fea_weight
        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('kd_reg_weight', None) is not None:
            reg_fea_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kd_reg_weight']
        else:
            reg_fea_weight = fea_weight

        feat_totall_loss = fea_loss * fea_weight + cls_fea_loss * cls_fea_weight + reg_fea_loss * reg_fea_weight



        fea_loss = fea_loss / student_feature.shape[0]
        tb_dict = {
            'rpn_spatial_feature_loss': fea_loss.item(),
            'rpn_cls_fea_loss': cls_fea_loss.item(),
            'rpn_reg_fea_loss': reg_fea_loss.item(),
            'rpn_feat_totall_loss': feat_totall_loss.item(),
        }
        return fea_loss, tb_dict


    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        kd_cls_loss, tb_dict_cls_teach = self.get_kd_cls_loss()
        kd_reg_loss, tb_dict_reg_teach = self.get_kd_reg_loss()
        rpn_loss = cls_loss + box_loss + kd_cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kd_hard_cls_weight'] \
                    + kd_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kd_hard_reg_weight']
        tb_dict.update(tb_dict_box)
        tb_dict.update(tb_dict_cls_teach)

        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('kd_fea_weight', None) is not None:
            fea_loss, tb_fea_dict = self.get_hint_loss()
            rpn_loss += fea_loss
            tb_dict.update(tb_fea_dict)

        if self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('kd_con_weight', None) is not None:
            con_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['kd_con_weight']
            con_loss, tb_con_dict = self.consistency_loss()
            rpn_loss += con_loss[0] * con_weight
            tb_dict.update(tb_con_dict)

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict



    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_temp  = self.cls_stems(spatial_features_2d)
        reg_temp  = self.reg_stems(spatial_features_2d)

        cls_preds = self.conv_cls(cls_temp)
        box_preds = self.conv_box(reg_temp)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]


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

            teacher_dict = self.assign_targets(
                gt_boxes=data_dict['teacher_box']
            )
            self.forward_ret_dict['cls_preds'] = cls_preds
            self.forward_ret_dict['box_preds'] = box_preds
            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
            self.forward_ret_dict['student_feature'] = spatial_features_2d
            self.forward_ret_dict['student_cls_temp'] = cls_temp
            self.forward_ret_dict['student_reg_temp'] = reg_temp
            self.knowledge_forward_rect.update(teacher_dict)
            self.knowledge_forward_rect['teacher_feature']  = data_dict['teacher_feature']
            self.knowledge_forward_rect['teacher_cls_pred'] = data_dict['teacher_cls_pred'] 
            self.knowledge_forward_rect['teacher_head_cls_temp'] = data_dict['teacher_cls_feature']
            self.knowledge_forward_rect['teacher_head_reg_temp'] = data_dict['teacher_reg_feature']
            self.knowledge_forward_rect['teacher_dir_pred'] = data_dict['teacher_dir_pred']
            self.knowledge_forward_rect['teacher_box_pred'] = data_dict['teacher_box_pred']

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            self.forward_ret_dict['batch_cls_preds'] = batch_cls_preds
            self.forward_ret_dict['batch_box_preds'] = batch_box_preds
            self.forward_ret_dict['batch_dir_cls_preds'] = dir_cls_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
