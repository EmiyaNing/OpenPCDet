import torch 
import copy 
from numpy.lib.function_base import disp
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms


class FUSIONNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, model_1, model_2):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.model1 = model_1
        self.model2 = model_2
        

    def forward(self, batch_dict):
        temp_batch = copy.deepcopy(batch_dict)
        _, _, = self.model1(batch_dict)
        _, _, _= self.model2(temp_batch)
        post_batch    = {}
        post_batch['batch_size']      = batch_dict['batch_size']
        post_batch['batch_box_preds'] = torch.cat([batch_dict['batch_box_preds'], temp_batch['batch_box_preds']], dim=1)
        post_batch['batch_cls_preds'] = torch.cat([batch_dict['batch_cls_preds'], temp_batch['batch_cls_preds']], dim=1)
        
        pred_dicts, recall_dicts = self.post_processing(post_batch)
        return pred_dicts, recall_dicts

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
