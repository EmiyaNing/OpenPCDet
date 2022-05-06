from operator import gt
from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
import numpy as np

from ..model_utils.model_nms_utils import class_agnostic_nms
class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            if cur_module.__class__.__name__ == 'DecoupleHeadThree':
                batch_dict['dense_head_cls_preds'] = batch_dict['batch_cls_preds']
                batch_dict['dense_head_box_preds'] = batch_dict['batch_box_preds']

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if self.model_cfg.TEACHER:
                for i in range(len(pred_dicts)):
                    pred_dicts[i]['teacher_cls_preds'] = batch_dict['dense_head_cls_preds'][i]
                    pred_dicts[i]['teacher_box_preds'] = batch_dict['dense_head_box_preds'][i]
                    pred_dicts[i]['teacher_feature'] = batch_dict['spatial_features_2d'][i]
                    pred_dicts[i]['teacher_cls_temp'] = batch_dict['kd_cls_temp'][i]
                    pred_dicts[i]['teacher_reg_temp'] = batch_dict['kd_reg_temp'][i]
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot


def decode_filp(gt_boxes, enable_list):
    for i in range(gt_boxes.shape[0]):
        noise_box = gt_boxes[i]
        if enable_list[i]:
            noise_box[:, 1] = -noise_box[:, 1]
            noise_box[:, 6] = -noise_box[:, 6]
    return gt_boxes

def decode_global_rotation(gt_boxes, rotation_list):
    for i in range(gt_boxes.shape[0]):
        noise_box = gt_boxes[i]
        noise_box[:, 0:3] = rotate_points_along_z(noise_box[:, 0:3].unsqueeze(0), -rotation_list[i].unsqueeze(0))[0]
        noise_box[:, 6] -= rotation_list[i]
    return gt_boxes

def decode_scaling(gt_boxes, scale_range):
    for i in range(gt_boxes.shape[0]):
        gt_boxes[:, :6] /= scale_range[i]
    return gt_boxes



class VoxelRCNN_TTA(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        if self.training:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
                if cur_module.__class__.__name__ == 'DecoupleHeadThree':
                    batch_dict['dense_head_cls_preds'] = batch_dict['batch_cls_preds']
                    batch_dict['dense_head_box_preds'] = batch_dict['batch_box_preds']
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            total_dict = {}
            for index, dicts in enumerate(batch_dict):
                for cur_module in self.module_list:
                    dicts = cur_module(dicts)
                
                if index == 0:
                    total_dict.update(dicts)
                elif index == 1:
                    boxes_list = decode_filp(dicts['batch_box_preds'], dicts['filp_enable'])
                    total_dict['batch_box_preds'] = torch.cat([total_dict['batch_box_preds'], boxes_list], dim=1)
                    total_dict['batch_cls_preds'] = torch.cat([total_dict['batch_cls_preds'], dicts['batch_cls_preds']], dim=1)
                elif index == 2:
                    boxes_list = decode_global_rotation(dicts['batch_box_preds'], dicts['rotation_noise'])
                    total_dict['batch_box_preds'] = torch.cat([total_dict['batch_box_preds'], boxes_list], dim=1)
                    total_dict['batch_cls_preds'] = torch.cat([total_dict['batch_cls_preds'], dicts['batch_cls_preds']], dim=1)
                elif index == 3:
                    boxes_list = decode_scaling(dicts['batch_box_preds'], dicts['scale_noise'])
                    total_dict['batch_box_preds'] = torch.cat([total_dict['batch_box_preds'], boxes_list], dim=1)
                    total_dict['batch_cls_preds'] = torch.cat([total_dict['batch_cls_preds'], dicts['batch_cls_preds']], dim=1)

            pred_dicts, recall_dicts = self.post_processing(total_dict)

        
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):

            box_preds = batch_dict['batch_box_preds'][index]
            cls_preds = batch_dict['batch_cls_preds'][index]
            cls_preds = torch.sigmoid(cls_preds)

            cls_scores, label_preds = torch.max(cls_preds, dim=-1)
            label_preds   = label_preds + 1

            src_box_preds = box_preds
            nms_scores   = cls_scores


            selected, selected_scores = class_agnostic_nms(
                box_scores=nms_scores, box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

            if post_process_cfg.OUTPUT_RAW_SCORE:
                raise NotImplementedError

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_cls_scores': cls_preds[selected],
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict
