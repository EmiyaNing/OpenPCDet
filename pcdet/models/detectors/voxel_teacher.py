
from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn

class VoxelTEACHRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            if cur_module.__class__.__name__ == 'DecoupleHeadThree':
                batch_dict['dense_head_cls_preds'] = batch_dict['batch_cls_preds']
                batch_dict['dense_head_box_preds'] = batch_dict['batch_box_preds']
                batch_dict['dense_head_dir_preds'] = batch_dict['dir_cls_preds']


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if self.model_cfg.TEACHER:
                teacher_dict = {}
                teacher_dict['teacher_dir_pred']  = batch_dict['dense_head_dir_preds']
                teacher_dict['teacher_cls_pred']  = batch_dict['dense_head_cls_preds']
                teacher_dict['teacher_feature']   = batch_dict['spatial_features_2d']
                teacher_dict['teacher_bev_feature']= batch_dict['spatial_features']
                teacher_dict['teacher_box_pred']  = batch_dict['dense_head_box_preds']
                teacher_dict['teacher_cls_feature'] = batch_dict['kd_cls_temp']
                teacher_dict['teacher_reg_feature'] = batch_dict['kd_reg_temp']
                return pred_dicts, recall_dicts, teacher_dict
            else:
                return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict



class VoxelTEACHRCNNV2(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            if cur_module.__class__.__name__ == 'DecoupleHeadThree':
                batch_dict['dense_head_cls_preds'] = batch_dict['batch_cls_preds']
                batch_dict['dense_head_box_preds'] = batch_dict['batch_box_preds']
                batch_dict['dense_head_dir_preds'] = batch_dict['dir_cls_preds']


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if self.model_cfg.TEACHER:
                teacher_dict = {}
                teacher_dict['teacher_dir_pred']  = batch_dict['dense_head_dir_preds']
                teacher_dict['teacher_cls_pred']  = batch_dict['dense_head_cls_preds']
                teacher_dict['teacher_feature']   = batch_dict['spatial_features_2d']
                teacher_dict['teacher_bev_feature']= batch_dict['spatial_features']
                teacher_dict['teacher_box_pred']  = batch_dict['dense_head_box_preds']
                return pred_dicts, recall_dicts, teacher_dict
            else:
                return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
