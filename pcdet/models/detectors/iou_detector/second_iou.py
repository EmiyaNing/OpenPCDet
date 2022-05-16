import torch
import torch.nn as nn
import torch.nn.functional as F
from ..detector3d_template import Detector3DTemplate
from ....ops.iou3d_nms import iou3d_nms_utils 
from ...model_utils.model_nms_utils import class_agnostic_nms
from ...model_utils.meter_utils import AverageMeter

class IOU_SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.forward_ret_dict = {}

    def get_iou_loss(self):
        batch_size          = self.forward_ret_dict['batch_size']
        one_stage_iou_preds = self.forward_ret_dict['one_stage_iou']
        #one_stage_label     = self.forward_ret_dict['one_stage_label']
        one_stage_boxs      = self.forward_ret_dict['one_stage_boxs'].clone().detach()
        ground_truth_box    = self.forward_ret_dict['gt_boxes']
        #positives           = one_stage_label > 0
        iou3d_loss          = 0
        iou_positive_count  = 0
        iou_stastic_1       = 0
        iou_stastic_3       = 0
        for bs_idx in range(batch_size):
            batch_gt_box= ground_truth_box[bs_idx]
            #pos_mask    = positives[bs_idx]
            pred_boxes  = one_stage_boxs[bs_idx]
            iou_preds   = one_stage_iou_preds[bs_idx]
            iou_positive_count += (iou_preds > 0).int().sum().item()
            iou_stastic_1      += (iou_preds > 0.1).int().sum().item()
            iou_stastic_3      += (iou_preds > 0.3).int().sum().item()
            iou_ground,_= iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, batch_gt_box[:, 0:7]).max(dim=-1, keepdim=True)
            iou3d_loss  += F.smooth_l1_loss(iou_preds, iou_ground)

        iou3d_loss          /= batch_size
        tb_dict = {
            'iou3d_loss': iou3d_loss.item(),
            'iou_positive_predict_count': iou_positive_count,
            'iou_predict_biger_than_0.1': iou_stastic_1,
            'iou_predict_biger_than_0.3': iou_stastic_3
        }
        return iou3d_loss, tb_dict


    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        if self.training:
            self.forward_ret_dict['one_stage_boxs'] = batch_dict['batch_box_preds']
            self.forward_ret_dict['one_stage_cls']  = batch_dict['batch_cls_preds']
            self.forward_ret_dict['one_stage_iou']  = batch_dict['batch_box_ious']
            self.forward_ret_dict['gt_boxes']       = batch_dict['gt_boxes']
            #self.forward_ret_dict['one_stage_label']= batch_dict['box_cls_labels']
            self.forward_ret_dict['batch_size']     = batch_dict['batch_size']
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        iou_loss, tb_iou_dict = self.get_iou_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        tb_dict.update(tb_iou_dict)

        loss = loss_rpn + iou_loss
        return loss, tb_dict, disp_dict


    '''def post_processing(self, batch_dict):
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
            batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_box_ious'][batch_mask]
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)

            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            label_preds = label_preds + 1


            keep_mask   = cls_preds > post_process_cfg.SCORE_THRESH
            box_preds   = box_preds[keep_mask]
            iou_preds   = iou_preds[keep_mask]
            cls_preds   = cls_preds[keep_mask]


            nms_scores  = cls_preds
            ### We may could add di-nms in here....
            #iou_preds   = (iou_preds.squeeze() + 1) * 0.5
            #nms_scores  *= torch.pow(iou_preds, 4)


            selected, selected_scores = class_agnostic_nms(
                box_scores=nms_scores, box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

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
                'pred_iou_scores': iou_preds[selected]
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict'''
