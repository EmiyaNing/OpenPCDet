import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .detector3d_template import Detector3DTemplate
from .. import roi_heads


class AverageMeter():
    """ Meter for monitoring losses"""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0
        self.reset()

    def reset(self):
        """reset all values to zeros"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        """update avg by val and n, where val is the avg of n values"""
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Voxel_CT3D(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology.append('sub_roi_head')
        self.module_list = self.build_networks()
        self.forward_ret_dict = {}
        self.sub_consistency_meter  = AverageMeter()
        self.main_consistency_meter = AverageMeter()
        self.voxel_head_rcnn_cls_meter = AverageMeter()
        self.ct3d_head_rcnn_cls_meter  = AverageMeter()

    def build_sub_roi_head(self, model_info_dict):
        if self.model_cfg.get('SUB_ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.SUB_ROI_HEAD.NAME](
            model_cfg=self.model_cfg.SUB_ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_class=self.num_class if not self.model_cfg.SUB_ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, batch_dict):
        if self.training:
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.map_to_bev_module(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)
            #copy_dict  = copy.deepcopy(batch_dict)
            batch_dict = self.roi_head(batch_dict)
            self.forward_ret_dict['stage_one_box'] = batch_dict['stage_one_box']
            self.forward_ret_dict['stage_one_cls'] = batch_dict['stage_one_cls']
            self.forward_ret_dict['cur_epoch']     = batch_dict['cur_epoch']

            main_batch_cls_preds, main_batch_box_preds = self.roi_head.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=batch_dict['rcnn_cls'], box_preds=batch_dict['rcnn_reg']
            )
            self.forward_ret_dict['main_stage_two_box'] = main_batch_box_preds
            self.forward_ret_dict['main_stage_two_cls'] = main_batch_cls_preds
            self.forward_ret_dict['main_stage_two_labels'] = batch_dict['roi_labels']

            batch_dict  = self.sub_roi_head(batch_dict)
            sub_batch_cls_preds, sub_batch_box_preds = self.sub_roi_head.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=batch_dict['rcnn_cls'], box_preds=batch_dict['rcnn_reg']
            )

            self.forward_ret_dict['sub_stage_two_box'] = sub_batch_box_preds
            self.forward_ret_dict['sub_stage_two_cls'] = sub_batch_cls_preds
            self.forward_ret_dict['sub_stage_two_labels'] = batch_dict['roi_labels']

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
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss_subrcnn, tb_sub_dict = self.sub_roi_head.get_loss(tb_dict)

            loss = loss_rpn + loss_rcnn + loss_subrcnn
        else:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss_subrcnn, tb_sub_dict = self.sub_roi_head.get_loss(tb_dict)
            self.ct3d_head_rcnn_cls_meter.update(tb_sub_dict['rcnn_loss_cls'])
            self.voxel_head_rcnn_cls_meter.update(tb_dict['rcnn_loss_cls'])
            loss_main_kd, tb_mainkd_dict = self.get_main_roihead_self_distillation_loss()
            loss_sub_kd,  tb_subkd_dict  = self.get_sub_roihead_self_distillation_loss()
            loss = loss + loss_rpn + loss_rcnn + loss_subrcnn + loss_main_kd * self.model_cfg.SELFWEIGHT + loss_sub_kd * self.model_cfg.SELFWEIGHT
            tb_dict.update(tb_mainkd_dict)
            tb_dict.update(tb_subkd_dict)
            tb_dict['mean_ct3d_rcnn_cls'] = self.ct3d_head_rcnn_cls_meter.avg
            tb_dict['mean_voxel_rcnn_cls']= self.voxel_head_rcnn_cls_meter.avg
        
        return loss, tb_dict, disp_dict

    def get_main_roihead_self_distillation_loss(self):
        # get parameters.....
        #cost_function = nn.BCELoss(reduction='none')
        cost_function = nn.KLDivLoss(reduction='none')
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
        cls_loss = cost_function(matched_student_cls_preds.log(), one_hot_targets)
        cls_loss = (cls_loss * matched_student_mask).sum() / (num_teacher_box * batch_sz)
        self.sub_consistency_meter.update(cls_loss.item())
        tb_dict = {
            'Self-kd-main-cls-loss': cls_loss.item(),
            'Self-kd-main-cls-mean-loss': self.sub_consistency_meter.avg,
        }   
        return cls_loss, tb_dict


    def get_sub_roihead_self_distillation_loss(self):
        # get parameters.....
        #cost_function = nn.BCELoss(reduction='none')
        cost_function = nn.KLDivLoss(reduction='none')
        epoch         = self.forward_ret_dict['cur_epoch']
        stage_one_cls = F.sigmoid(self.forward_ret_dict['stage_one_cls'])
        stage_one_box = self.forward_ret_dict['stage_one_box']
        stage_two_cls = F.sigmoid(self.forward_ret_dict['sub_stage_two_cls'])
        stage_two_box = self.forward_ret_dict['sub_stage_two_box']
        stage_two_label = self.forward_ret_dict['sub_stage_two_labels']
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
        cls_loss = cost_function(matched_student_cls_preds.log(), one_hot_targets)
        cls_loss = (cls_loss * matched_student_mask).sum() / (num_teacher_box * batch_sz)
        self.main_consistency_meter.update(cls_loss.item())
        tb_dict = {
            'Self-kd-sub-cls-loss': cls_loss.item(),
            'Self-kd-sub-cls-mean-loss': self.main_consistency_meter.avg,
        }   
        return cls_loss, tb_dict
