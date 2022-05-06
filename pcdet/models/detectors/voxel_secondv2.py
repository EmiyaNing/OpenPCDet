from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from ..model_utils.meter_utils import AverageMeter


class Self_VoxelSECOND(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        if self.training:
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            if not self.model_cfg.SELFKD:
                batch_dict = self.dense_head(batch_dict)
                batch_dict = self.roi_head(batch_dict)
            else:
                batch_dict = self.dense_head(batch_dict)
                batch_dict = self.roi_head(batch_dict)
                self.forward_ret_dict['stage_one_cls'] = batch_dict['stage_one_cls']
                self.forward_ret_dict['stage_one_box'] = batch_dict['stage_one_box']
                self.forward_ret_dict['stage_two_cls'] = batch_dict['batch_cls_preds']
                self.forward_ret_dict['stage_two_box'] = batch_dict['batch_box_preds']
                self.forward_ret_dict['stage_two_label'] = batch_dict['roi_labels']
                self.forward_ret_dict['cur_epoch'] = batch_dict['cur_epoch']
            
            if self.model_cfg.ATTENTION_DISTILLATION:
                self.forward_ret_dict['attention_group1'] = batch_dict['attention_group1']
                self.forward_ret_dict['attention_group2'] = batch_dict['attention_group2']
                self.forward_ret_dict['attention_label']  = batch_dict['attention_group3']

                
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
            #batch_dict = self.roi_head(batch_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0 
        if not self.model_cfg.SELFKD:  
            loss_rpn, tb_dict  = self.dense_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss + loss_rpn + loss_rcnn
        else:
            loss_rpn, tb_dict   = self.dense_head.get_loss()
            loss_rcnn, tb_dict  = self.roi_head.get_loss(tb_dict)
            loss_kd, tb_kd_dict = self.get_self_distillation_loss() 
            loss = loss + loss_rpn + loss_rcnn + loss_kd
            tb_dict.update(tb_kd_dict) 
        
        if self.model_cfg.ATTENTION_DISTILLATION:
            loss_attn, tb_attn_dict  = self.get_atteintion_self_distillation_loss()
            loss += loss_attn
            tb_dict.update(tb_attn_dict)

        return loss, tb_dict, disp_dict

    def get_self_distillation_loss(self):
        # get parameters.....
        epoch         = self.forward_ret_dict['cur_epoch']
        stage_one_cls = F.sigmoid(self.forward_ret_dict['stage_one_cls'])
        stage_one_box = self.forward_ret_dict['stage_one_box']
        stage_two_cls = F.sigmoid(self.forward_ret_dict['stage_two_cls'])
        stage_two_box = self.forward_ret_dict['stage_two_box']
        stage_two_label = self.forward_ret_dict['stage_two_label']
        batch_sz = stage_one_box.shape[0]

        # filter once by confidence
        one_confidence, _ = torch.max(stage_one_cls, dim=-1)
        two_confidence, _ = torch.max(stage_two_cls, dim=-1)
        if epoch < 10:
            THRESH = (epoch + 1) / 20 + 0.05
        else:
            THRESH = 0.7
        one_mask = one_confidence > THRESH
        two_mask = two_confidence > THRESH
        stage_one_cls, stage_one_box = stage_one_cls[one_mask], stage_one_box[one_mask]
        if stage_one_cls.shape == 0:
            tb_dict = {
                'Self-kd-cls-loss': 0
            }
            return 0, tb_dict
        stage_two_cls, stage_two_box, stage_two_label = stage_two_cls[two_mask], stage_two_box[two_mask], stage_two_label[two_mask]
        one_hot_targets = torch.zeros(
            *list(stage_two_cls.squeeze(-1).shape), self.num_class + 1, dtype=stage_two_cls.dtype, device=stage_two_cls.device
        )
        one_hot_targets.scatter_(-1, stage_two_label.unsqueeze(dim=-1).long(), stage_two_cls)
        one_hot_targets = one_hot_targets[:, 1:]
        # soft the one_hot_targets
        zero_mask = one_hot_targets == 0
        one_hot_targets[zero_mask] = 0.01

        # match the one_stage_box and two_stage_box
        num_teacher_box = stage_two_box.shape[0]

        teacher_centers = stage_two_box[:, :3]
        student_centers = stage_one_box[:, :3]

        with torch.no_grad():
            teacher_class = stage_two_label.unsqueeze(-1)
            student_class = torch.max(stage_one_cls, dim=-1, keepdim=True)[1]
            not_same_class = (teacher_class != student_class.T).float() # [Nt, Ns]
            MAX_DISTANCE = 1000000
            dist = teacher_centers[:, None, :] - student_centers[None, :, :] # [Nt, Ns, 3]
            dist = (dist ** 2).sum(-1) # [Nt, Ns]
            dist += not_same_class * MAX_DISTANCE # penalty on different classes
            student_dist_of_teacher, student_index_of_teacher = dist.min(1) # [Nt]
            # different from standard sess, we only consider distance<1m as matching
            MATCHED_DISTANCE = 1
            matched_student_mask = (student_dist_of_teacher < MATCHED_DISTANCE).float().unsqueeze(-1) # [Nt, 1]

        matched_student_cls_preds = stage_one_cls[student_index_of_teacher]
        cls_loss = F.mse_loss(matched_student_cls_preds, one_hot_targets, reduction='none')
        cls_loss = (cls_loss * matched_student_mask).sum() / (num_teacher_box * batch_sz)
        tb_dict = {
            'Self-kd-cls-loss': cls_loss.item()
        }   
        return cls_loss, tb_dict

    def get_atteintion_self_distillation_loss(self):
        cost_function = nn.MSELoss()
        attention_label = self.forward_ret_dict['attention_label']
        attnetion_group1= self.forward_ret_dict['attention_group1']
        attention_group2= self.forward_ret_dict['attention_group2']
        loss = self.model_cfg.ATTENTION_WEIGHT * (cost_function(attention_label, attnetion_group1) + cost_function(attention_label, attention_group2))
        tb_dict = {
            'attention_loss': loss.item()
        } 
        return loss, tb_dict
