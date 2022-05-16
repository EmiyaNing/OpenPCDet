import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .detector3d_template import Detector3DTemplate
from .. import roi_heads
from ..model_utils import centernet_utils
from ..model_utils.meter_utils import AverageMeter
from ...utils import loss_utils


class Voxel_Second_Subnet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.distill_cost_function = nn.KLDivLoss(reduction='none')
        self.forward_ret_dict = {}
        self.main_consistency_meter = AverageMeter()
        self.voxel_head_rcnn_cls_meter = AverageMeter()
        self.tempture          = self.model_cfg.TEMPTURE
        self.point_cloud_range = self.dataset.point_cloud_range
        self.voxel_size        = self.dataset.voxel_size

        self.foreground_costfunction = loss_utils.FocalLossCenterNet()

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y


    def assign_subnet_single_target(self, num_class, gt_boxes, feature_map_size, feature_map_stride, \
                             num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        target_map = gt_boxes.new_zeros(num_class, feature_map_size[1], feature_map_size[0], device=gt_boxes.device)
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(target_map[cur_class_id], center[k], radius[k].item())

        return target_map


    def assign_subnet_batch_target(self, gt_boxes, feature_map_size=None, **kwargs):
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        batch_size       = gt_boxes.shape[0]
        target_map_list  = []

        target_assigner_cfg = self.model_cfg.FOREGROUND_ASSIGNER_CONFIG

        for bs_idx in range(batch_size):
            cur_gt_boxes = gt_boxes[bs_idx]
            target_map   = self.assign_subnet_single_target(
                num_class=self.num_class, gt_boxes=cur_gt_boxes.cpu(), feature_map_size=feature_map_size,
                feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                min_radius=target_assigner_cfg.MIN_RADIUS,
            )
            target_map_list.append(target_map)

        subnet_target = torch.stack(target_map_list, dim=0)
        return subnet_target

    def get_sub_foreground_loss(self):
        foreground_res = self.sigmoid(self.forward_ret_dict['subnet_result'])
        foreground_tar = self.forward_ret_dict['subnet_target'].to(foreground_res.device)
        batch_size     = foreground_res.shape[0]

        sub_loss       = self.foreground_costfunction(foreground_res, foreground_tar)
        sub_loss       /= batch_size
        sub_weight     = self.get_sub_task_weights()
        sub_loss       = sub_weight * sub_loss
        tb_dict = {
            'spatial_fourground_segment_loss': sub_loss.item(),
        }
        return sub_loss, tb_dict

    def get_sub_task_weights(self):
        cur_epoch   = self.forward_ret_dict['cur_epoch']
        total_epoch = self.forward_ret_dict['total_epoch']
        sub_weights = self.model_cfg.SUB_WEIGHTS * (cur_epoch / total_epoch)
        return sub_weights



    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        if self.training:
            batch_dict = self.roi_head(batch_dict)
            self.forward_ret_dict['stage_one_box'] = batch_dict['stage_one_box']
            self.forward_ret_dict['stage_one_cls'] = batch_dict['stage_one_cls']
            self.forward_ret_dict['cur_epoch']     = batch_dict['cur_epoch']
            self.forward_ret_dict['total_epoch']   = batch_dict['total_epoch']

            subnet_target = self.assign_subnet_batch_target(batch_dict['gt_boxes'], feature_map_size=batch_dict['spatial_features'].size()[2:], 
            feature_map_stride=batch_dict.get('spatial_features_stride', None)
            )

            self.forward_ret_dict['subnet_target'] = subnet_target
            self.forward_ret_dict['subnet_result'] = batch_dict['sub_foreground_segment_result']

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
            loss_sub, tb_sub_dict = self.get_sub_foreground_loss()
            tb_dict.update(tb_sub_dict)
            self.voxel_head_rcnn_cls_meter.update(tb_dict['rcnn_loss_cls'])


            L_teacher_cls = tb_dict['rcnn_loss_cls']
            L_student_cls = tb_dict['rpn_loss_cls']


            dynamic_kd_weight = (1 - math.exp(- L_student_cls / L_teacher_cls))

            loss_main_kd, tb_mainkd_dict = self.get_main_roihead_self_distillation_loss()
            loss = loss + loss_rpn + loss_rcnn+ loss_main_kd * dynamic_kd_weight * self.model_cfg.SELFWEIGHT + loss_sub
            tb_dict.update(tb_mainkd_dict)
            tb_dict['mean_voxel_rcnn_cls']= self.voxel_head_rcnn_cls_meter.avg
        
        return loss, tb_dict, disp_dict

    def get_main_roihead_self_distillation_loss(self):
        epoch         = self.forward_ret_dict['cur_epoch']
        stage_one_cls = F.sigmoid(self.forward_ret_dict['stage_one_cls'])
        stage_one_box = self.forward_ret_dict['stage_one_box']
        stage_two_cls = F.sigmoid(self.forward_ret_dict['main_stage_two_cls'])
        stage_two_box = self.forward_ret_dict['main_stage_two_box']
        stage_two_label = self.forward_ret_dict['main_stage_two_labels']
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
        one_hot_targets[zero_mask] = 0.0001

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
        cls_loss = self.distill_cost_function(matched_student_cls_preds.log(), one_hot_targets)
        cls_loss = (cls_loss * matched_student_mask).sum() / (num_teacher_box * batch_sz)
        self.main_consistency_meter.update(cls_loss.item())
        tb_dict = {
            'Self-kd-main-cls-loss': cls_loss.item(),
            'Self-kd-main-cls-mean-loss': self.main_consistency_meter.avg,
        }   
        return cls_loss, tb_dict

