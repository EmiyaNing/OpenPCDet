from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VoxelSECOND(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        if self.training:
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

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

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        self_centers_loss, self_size_loss, self_cls_loss = self.get_self_distillation_loss(batch_dict)
        self_consistency_loss = self_centers_loss * self.model_cfg.SELF_DISTILLATION.CENTER_WEIGHT + \
                                self_size_loss * self.model_cfg.SELF_DISTILLATION.SIZE_WEIGHT + self_cls_loss * self.model_cfg.SELF_DISTILLATION.CLS_WEIGHT
        con_weight = self.model_cfg.SELF_DISTILLATION.CON_WEIGHT * sigmoid_rampup(batch_dict['cur_epoch'], -1, 48)    

        loss = loss + loss_rpn + loss_rcnn + self_consistency_loss * con_weight
        tb_dict['self_kd_center_loss'] = self_centers_loss.item()
        tb_dict['self_kd_size_loss']   = self_size_loss.item()
        tb_dict['self_kd_cls_loss']    = self_cls_loss.item()
        tb_dict['self_kd_consistency_loss'] = self_consistency_loss.item()
        return loss, tb_dict, disp_dict

    def get_self_distillation_loss(self, data_dict):
        stage_one_cls = data_dict['roi_cls']
        stage_one_reg = data_dict['rois_copy']
        stage_two_cls = data_dict['batch_cls_preds']
        stage_two_reg = data_dict['batch_box_preds']
        batch_size = stage_one_cls.shape[0]
        center_losses, size_losses, cls_losses = [], [], []
        batch_normalizer = 0
        for idx in range(batch_size):
            teacher_cls = stage_two_cls[idx]
            teacher_box = stage_two_reg[idx]
            student_cls = stage_one_cls[idx]
            student_box = stage_one_reg[idx]
            num_teacher_boxes = teacher_box.shape[0]
            num_student_boxes = student_box.shape[0]
            if num_teacher_boxes == 0 or num_student_boxes == 0:
                batch_normalizer += 1
                continue
            teacher_centers, teacher_size, teacher_rot = teacher_box[:, :3], teacher_box[:, 3:6], teacher_box[:, [6]]
            student_centers, student_size, student_rot = student_box[:, :3], student_box[:, 3:6], student_box[:, [6]]

            with torch.no_grad():
                teacher_class  = torch.max(teacher_cls, dim=-1, keepdim=True)[1]
                student_class  = torch.max(student_cls, dim=-1, keepdim=True)[1]
                not_same_class = (teacher_class != student_class.T).float() 
                MAX_DISTANCE = 1000000
                dist = teacher_centers[:, None, :] - student_centers[None, :, :]
                dist = (dist ** 2).sum(-1)
                dist += not_same_class * MAX_DISTANCE
                student_dist_of_teacher, student_index_of_teacher = dist.min(1) # [Nt]
                teacher_dist_of_student, teacher_index_of_student = dist.min(0) # [Ns]
                MATCHED_DISTANCE = 1
                matched_teacher_mask = (teacher_dist_of_student < MATCHED_DISTANCE).float().unsqueeze(-1) # [Ns, 1]
                matched_student_mask = (student_dist_of_teacher < MATCHED_DISTANCE).float().unsqueeze(-1) # [Nt, 1]
            
            matched_teacher_centers = teacher_centers[teacher_index_of_student] # [Ns, :]
            matched_student_centers = student_centers[student_index_of_teacher] # [Nt, :]

            matched_student_sizes = student_size[student_index_of_teacher] # [Nt, :]
            matched_student_cls_preds = student_cls[student_index_of_teacher] # [Nt, :]

            center_loss = (((student_centers - matched_teacher_centers) * matched_teacher_mask).abs().sum()
                       + ((teacher_centers - matched_student_centers) * matched_student_mask).abs().sum()) \
                      / (num_teacher_boxes + num_student_boxes)
            size_loss = F.mse_loss(matched_student_sizes, teacher_size, reduction='none')
            size_loss = (size_loss * matched_student_mask).sum() / num_teacher_boxes

            # kl_div is not feasible, since we use sigmoid instead of softmax for class prediction
            # cls_loss = F.kl_div(matched_student_cls_preds.log(), teacher_cls_preds, reduction='none')
            cls_loss = F.mse_loss(matched_student_cls_preds, teacher_cls, reduction='none') # use mse loss instead
            cls_loss = (cls_loss * matched_student_mask).sum() / num_teacher_boxes

            center_losses.append(center_loss)
            size_losses.append(size_loss)
            cls_losses.append(cls_loss)
            batch_normalizer += 1
        return sum(center_losses)/batch_normalizer, sum(size_losses)/batch_normalizer, sum(cls_losses)/batch_normalizer
            
def sigmoid_rampup(current, rampup_start, rampup_end):
    assert rampup_start <= rampup_end
    if current < rampup_start:
        return 0
    elif (current >= rampup_start) and (current < rampup_end):
        rampup_length = max(rampup_end, 0) - max(rampup_start, 0)
        if rampup_length == 0: # no rampup
            return 1
        else:
            phase = 1.0 - (current - max(rampup_start, 0)) / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    elif current >= rampup_end:
        return 1
    else:
        raise Exception('Impossible condition for sigmoid rampup')