from numpy.lib.function_base import disp
from .detector3d_template import Detector3DTemplate

class SESSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            if not batch_dict['is_ema']:
                loss, tb_dict, disp_dict = self.get_training_loss()

                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                ema_dicts = {}
                ema_dicts['ema_feature'] = batch_dict['spatial_features_2d']
                ema_dicts['ema_cls_preds'] = batch_dict['batch_cls_preds']
                ema_dicts['ema_box_preds'] = batch_dict['batch_box_preds']
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict