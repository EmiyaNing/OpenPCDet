"""
    In this version Self_distillation framework, we detach reduent part, only retain the useful part.
    The remain parts include: sub_cls_heads, two-stage regularize heads. 
    Besides, we replace the linear rise smooth by the beta_smooth...
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..detector3d_template import Detector3DTemplate
from ...model_utils import centernet_utils
from ...model_utils.meter_utils import AverageMeter
from ....utils import loss_utils


class Voxel_SubHeadsF(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.distill_cost_function = nn.KLDivLoss(reduction='none')
        self.forward_ret_dict = {}
        self.main_consistency_meter = AverageMeter()
        self.sub_consistency_meter = AverageMeter()
        self.voxel_head_rcnn_cls_meter = AverageMeter()
        self.point_cloud_range = self.dataset.point_cloud_range
        self.voxel_size        = self.dataset.voxel_size
        self.dist_relation_cost= RkdDistance()
        self.angle_relation_cost = RKdAngle()

        self.sub_cls_heads     = nn.Sequential(
            nn.Conv2d(self.backbone_2d.input_channels, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.Conv2d(128, self.num_class * self.dense_head.num_anchors_per_location, 1, 1, padding=0),
        )
        self.sub_box_heads     = nn.Sequential(
            nn.Conv2d(self.backbone_2d.input_channels, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.Conv2d(128, 7 * self.dense_head.num_anchors_per_location, 1, 1, padding=0),
        )
        self.sub_dir_heads     = nn.Sequential(
            nn.Conv2d(self.backbone_2d.input_channels, 128, 1, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.SELU(),
            nn.Conv2d(128, self.dense_head.model_cfg.NUM_DIR_BINS * self.dense_head.num_anchors_per_location, 1, 1, padding=0),
        )




    def generate_one_hot_sublabels(self, box_cls_labels):
        cls_preds      = self.forward_ret_dict['sub_cls_preds']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        return one_hot_targets, cls_weights


    def get_sub_cls_loss(self):
        sub_result = self.forward_ret_dict['sub_cls_preds']
        batch_size = int(sub_result.shape[0])
        sub_result = sub_result.reshape(batch_size, -1, self.num_class)
        sub_target = self.forward_ret_dict['sub_cls_labels']
        sub_target, cls_weights = self.generate_one_hot_sublabels(sub_target)
        sub_loss   = self.dense_head.cls_loss_func(sub_result, sub_target, weights=cls_weights).sum()
        sub_loss   /= batch_size
        tb_dict = {
            'sub_class_loss': sub_loss.item(),
        }
        return sub_loss, tb_dict

    def get_sub_box_loss(self):
        sub_box_result = self.forward_ret_dict['sub_box_preds']
        sub_box_target = self.forward_ret_dict['sub_box_labels']
        sub_cls_target = self.forward_ret_dict['sub_cls_labels']
        batch_size     = self.forward_ret_dict['batch_size']

        positives = sub_cls_target > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        sub_box_result = sub_box_result.reshape(batch_size, -1, 7)
        sub_box_sin, sub_targets_sin = self.dense_head.add_sin_difference(sub_box_result, sub_box_target)
        sub_loc_loss   = self.dense_head.reg_loss_func(sub_box_sin, sub_targets_sin, weights=reg_weights)
        sub_loc_loss   = sub_loc_loss.sum() / batch_size
        tb_dict = {
            'sub_box_loss': sub_loc_loss.item(),
        }
        return sub_loc_loss, tb_dict

    def get_sub_dir_loss(self):
        sub_dir_preds  = self.forward_ret_dict['sub_dir_preds']
        box_cls_labels = self.forward_ret_dict['sub_cls_labels']
        box_reg_targets= self.forward_ret_dict['sub_box_labels']
        batch_size     = self.forward_ret_dict['batch_size']
        anchors        = self.dense_head.anchors
        if isinstance(anchors, list):
            anchors = torch.cat(anchors, dim=-3)
        
        anchors = anchors.reshape(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        dir_targets    = self.dense_head.get_direction_target(
            anchors, box_reg_targets,
            dir_offset = self.dense_head.model_cfg.DIR_OFFSET,
            num_bins   = self.dense_head.model_cfg.NUM_DIR_BINS
        )
        dir_logits     = sub_dir_preds.reshape(batch_size, -1, self.dense_head.model_cfg.NUM_DIR_BINS)
        
        positives   = box_cls_labels > 0
        dir_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        dir_weights /= torch.clamp(pos_normalizer, min=1.0)

        dir_loss    = self.dense_head.dir_loss_func(dir_logits, dir_targets, weights=dir_weights)
        dir_loss    = dir_loss.sum() / batch_size
        tb_dict = {
            'sub_dir_loss': dir_loss.item()
        }
        return dir_loss, tb_dict


    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        sub_feature= batch_dict['spatial_features']
        if self.training:
            batch_dict = self.dense_head(batch_dict)
            batch_dict = self.roi_head(batch_dict)

            batch_size = batch_dict['batch_size']
            self.forward_ret_dict['batch_size']    = batch_size
            self.forward_ret_dict['stage_one_box'] = batch_dict['stage_one_box']
            self.forward_ret_dict['stage_one_cls'] = batch_dict['stage_one_cls']
            self.forward_ret_dict['cur_epoch']     = batch_dict['cur_epoch']
            self.forward_ret_dict['total_epoch']   = batch_dict['total_epoch']
            self.forward_ret_dict['sub_cls_labels']= batch_dict['box_cls_labels']
            self.forward_ret_dict['sub_box_labels']= batch_dict['box_reg_targets']

            self.forward_ret_dict['sub_cls_preds'] = self.sub_cls_heads(sub_feature).permute(0, 2, 3, 1)
            self.forward_ret_dict['sub_box_preds'] = self.sub_box_heads(sub_feature).permute(0, 2, 3, 1)
            self.forward_ret_dict['sub_dir_preds'] = self.sub_dir_heads(sub_feature).permute(0, 2, 3, 1)


            main_batch_cls_preds, main_batch_box_preds = self.roi_head.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=batch_dict['rcnn_cls'], box_preds=batch_dict['rcnn_reg']
            )
            self.forward_ret_dict['main_stage_two_box'] = main_batch_box_preds
            self.forward_ret_dict['main_stage_two_cls'] = main_batch_cls_preds
            self.forward_ret_dict['main_stage_two_labels'] = batch_dict['roi_labels']

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            batch_dict = self.dense_head(batch_dict)
            #batch_dict = self.roi_head(batch_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts



    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        if not self.model_cfg.SELFKD:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

            loss = loss_rpn + loss_rcnn
        else:
            loss_rpn, tb_dict     = self.dense_head.get_loss()
            loss_rcnn, tb_dict    = self.roi_head.get_loss(tb_dict)
            loss_cls_sub, tb_sub_cls_dict = self.get_sub_cls_loss()
            tb_dict.update(tb_sub_cls_dict)
            self.voxel_head_rcnn_cls_meter.update(tb_dict['rcnn_loss_cls'])


            L_teacher_cls = tb_dict['rcnn_loss_cls']
            L_student_cls = tb_dict['rpn_loss_cls']


            dynamic_kd_weight = (1 - math.exp(- L_student_cls / L_teacher_cls))

            loss_main_kd, tb_mainkd_dict = self.get_main_roihead_self_distillation_loss()
            loss = loss + loss_rpn + loss_rcnn+ loss_main_kd * dynamic_kd_weight * self.model_cfg.SELFWEIGHT \
                    + loss_cls_sub * self.model_cfg.SELFWEIGHT
            tb_dict.update(tb_mainkd_dict)

            if self.model_cfg.get('SUBHEAD_DIR_LOSS', None) is not None:
                loss_sub_dir, tb_subdir_dict = self.get_sub_dir_loss()
                loss += loss_sub_dir * self.model_cfg.SUBHEAD_DIR_LOSS
                tb_dict.update(tb_subdir_dict)
            
            if self.model_cfg.get('SUBHEAD_BOX_LOSS', None) is not None:
                loss_sub_box, tb_subbox_dict = self.get_sub_box_loss()
                loss += loss_sub_box * self.model_cfg.SUBHEAD_BOX_LOSS
                tb_dict.update(tb_subbox_dict)

            if self.model_cfg.get('RELATION_LOSS_WEIGHT', None) is not None:
                loss_relation, tb_raltion_dict = self.get_relation_distillation_loss()
                loss += loss_relation * self.model_cfg.RELATION_LOSS_WEIGHT
                tb_dict.update(tb_raltion_dict)



            tb_dict['mean_voxel_rcnn_cls']= self.voxel_head_rcnn_cls_meter.avg


        return loss, tb_dict, disp_dict

    def get_main_roihead_self_distillation_loss(self):
        stage_one_cls = F.sigmoid(self.forward_ret_dict['stage_one_cls'])
        stage_one_box = self.forward_ret_dict['stage_one_box']
        stage_two_cls = F.sigmoid(self.forward_ret_dict['main_stage_two_cls'])
        stage_two_box = self.forward_ret_dict['main_stage_two_box']
        stage_two_label = self.forward_ret_dict['main_stage_two_labels']
        batch_sz    = stage_one_box.shape[0]

    

        # filter once by confidence
        one_confidence, _ = torch.max(stage_one_cls, dim=-1)
        two_confidence, _ = torch.max(stage_two_cls, dim=-1)


        con_thresh = 0.1

        one_mask = one_confidence > con_thresh
        two_mask = two_confidence > con_thresh
        stage_one_cls, stage_one_box = stage_one_cls[one_mask], stage_one_box[one_mask]
        if stage_one_cls.shape == 0:
            tb_dict = {
                'Self-kd-cls-loss': 0
            }
            return 0, tb_dict
        stage_two_cls, stage_two_box, stage_two_label = stage_two_cls[two_mask], stage_two_box[two_mask], stage_two_label[two_mask]


        linear_rise_weight = (1 - stage_two_cls) / (self.num_class - 1)



        one_hot_targets = torch.zeros(
            *list(stage_two_cls.squeeze(-1).shape), self.num_class + 1, dtype=stage_two_cls.dtype, device=stage_two_cls.device
        )
        one_hot_targets.scatter_(-1, stage_two_label.unsqueeze(dim=-1).long(), stage_two_cls)
        one_hot_targets = one_hot_targets[:, 1:].float()

        zero_mask = one_hot_targets == 0
        one_hot_targets[zero_mask] = linear_rise_weight.repeat(1, self.num_class - 1).reshape(-1)



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

        matched_student_cls_preds = stage_one_cls[student_index_of_teacher] # duiqi shape with teacher
        cls_loss = self.distill_cost_function(matched_student_cls_preds.log(), one_hot_targets)
        cls_loss = (cls_loss * matched_student_mask).sum() / (num_teacher_box * batch_sz)


        self.sub_consistency_meter.update(cls_loss.item())
        tb_dict = {
            'Self-kd-subheads-cls-loss': cls_loss.item(),
            'Self-kd-subheads-cls-mean-loss': self.sub_consistency_meter.avg,
        } 
        
        return cls_loss, tb_dict

    def get_relation_distillation_loss(self):
        stage_one_cls = F.sigmoid(self.forward_ret_dict['stage_one_cls'])
        stage_one_box = self.forward_ret_dict['stage_one_box']
        stage_two_cls = F.sigmoid(self.forward_ret_dict['main_stage_two_cls'])
        stage_two_box = self.forward_ret_dict['main_stage_two_box']
        stage_two_label = self.forward_ret_dict['main_stage_two_labels']
        batch_sz    = stage_one_box.shape[0]
        cls_loss    = 0

        compact_student = []
        compact_teacher = []

        for bs_idx in range(batch_sz):
            batch_stage_one_cls = stage_one_cls[bs_idx]
            batch_stage_one_box = stage_one_box[bs_idx]
            batch_stage_two_box = stage_two_box[bs_idx]
            batch_stage_two_cls = stage_two_cls[bs_idx]
            batch_stage_two_label = stage_two_label[bs_idx]

            one_confidence, _ = torch.max(batch_stage_one_cls, dim=-1)
            one_mask          = one_confidence > 0.1

            batch_stage_one_cls = batch_stage_one_cls[one_mask]
            batch_stage_one_box = batch_stage_one_box[one_mask]

            if batch_stage_one_cls.shape[0] == 0:
                continue


            linear_rise_weight  = (1 - batch_stage_two_cls) / (self.num_class - 1)

            one_hot_targets = torch.zeros(
                batch_stage_two_cls.shape[0], self.num_class + 1, dtype=batch_stage_two_cls.dtype, device=batch_stage_two_cls.device
            )


            one_hot_targets.scatter_(-1, batch_stage_two_label.unsqueeze(-1).long(), batch_stage_two_cls)
            one_hot_targets = one_hot_targets[:, 1:].float()

            zero_mask = one_hot_targets == 0
            one_hot_targets[zero_mask] = linear_rise_weight.repeat(1, self.num_class - 1).reshape(-1)

            # match the one_stage_box and two_stage_box
            num_teacher_box = batch_stage_two_box.shape[0]

            teacher_centers = batch_stage_two_box[:, :3]
            student_centers = batch_stage_one_box[:, :3]

            with torch.no_grad():
                teacher_class = batch_stage_two_label.unsqueeze(-1)
                student_class = torch.max(batch_stage_one_cls, dim=-1, keepdim=True)[1]
                not_same_class = (teacher_class != student_class.T).float() # [Nt, Ns]
                MAX_DISTANCE = 1000000
                dist = teacher_centers[:, None, :] - student_centers[None, :, :] # [Nt, Ns, 3]
                dist = (dist ** 2).sum(-1) # [Nt, Ns]
                dist += not_same_class * MAX_DISTANCE # penalty on different classes
                student_dist_of_teacher, student_index_of_teacher = dist.min(1) # [Nt]
                # different from standard sess, we only consider distance<1m as matching
                MATCHED_DISTANCE = 1
                matched_student_mask = (student_dist_of_teacher < MATCHED_DISTANCE).float().unsqueeze(-1)

            matched_student_cls_preds = batch_stage_one_cls[student_index_of_teacher] # duiqi shape with teacher
            matched_student_cls_preds = matched_student_cls_preds * matched_student_mask
            one_hot_targets           = one_hot_targets * matched_student_mask
            compact_student.append(matched_student_cls_preds)
            compact_teacher.append(one_hot_targets)

        compact_student = torch.stack(compact_student)
        compact_teacher = torch.stack(compact_teacher)

        angle_loss = self.angle_relation_cost(compact_student, compact_teacher)
        dist_loss  = self.dist_relation_cost(compact_student, compact_teacher)

        relation_loss = angle_loss * self.model_cfg.ANGLE_WEIGHT + dist_loss * self.model_cfg.DIST_WEIGHT
        tb_dict = {
            'relation_angle_loss': angle_loss.item(),
            'relation_dist_loss': dist_loss.item(),
            'relation_loss': relation_loss.item()
        }
        
        return cls_loss, tb_dict


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # batch x N x C
        # batch x batch x N x C

        student = student.permute(1, 0, 2)        # N, B, C
        teacher = teacher.permute(1, 0, 2)        # N, B, C
        with torch.no_grad():
            td = (teacher.unsqueeze(1) - teacher.unsqueeze(2))      # N, B, B, C
            norm_td = F.normalize(td, p=2, dim=3)                   # N, B, B, C
            t_angle = (norm_td @ norm_td.transpose(2, 3)).view(teacher.shape[0], -1)

        sd = (student.unsqueeze(1) - student.unsqueeze(2))
        norm_sd = F.normalize(sd, p=2, dim=3)
        #s_angle = torch.bmm(norm_sd, norm_sd.transpose(2, 3)).view(student.shape[0],-1)
        s_angle = (norm_sd @ norm_sd.transpose(2, 3)).view(student.shape[0], -1)
        
        s_angle = s_angle
        t_angle = t_angle 

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        '''
        student           A tensor with shape [batch_size, num_anchors, num_class]
        teacher           A tensor with shape [batch_size, num_anchors, num_class]
        '''
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)         # Now t_d.shape = num_anchors, batch_size, batch-size
            t_d = t_d.reshape(t_d.shape[0], -1)
            mean_td = t_d.mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        d = d.reshape(d.shape[0], -1)
        mean_d = d.mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


def pdist(teacher, squared=False, eps=1e-12):
    '''
        The input parameter:
            teacher            A tensor with shape [batch_size, num_anchors, num_class]
            squared            bool
            eps                a float represent the min value
        The return:
            A tensor with shape [Num_anchors, batch-size, batch-size]
    '''
    temp = teacher.permute(1, 0, 2)
    e_square = temp.pow(2).sum(dim=2)
    prod     = temp @ temp.permute(0, 2, 1)
    res      = (e_square.unsqueeze(2) + e_square.unsqueeze(1) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()
    
    res      = res.clone()
    res[:, range(len(teacher)), range(len(teacher))] = torch.zeros([teacher.shape[1], teacher.shape[0]], device= res.device)
    return res
