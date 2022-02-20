import torch
import torch.nn as nn
from numpy.lib.function_base import disp
from .detector3d_template import Detector3DTemplate


class SPVB_KD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        sigma = torch.randn(2, requires_grad=True)
        self.sigma = nn.Parameter(sigma)

    def forward(self, batch_dict, teacher_result):
        if self.training:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

            # Now the model need to caculate the knowledge loss of teacher and student.
            loss, tb_dict, disp_dict = self.get_training_loss(teacher_result)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, teacher_result):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_point': loss_point.item(),
            **tb_dict
        }
        return loss, tb_dict, disp_dict
