import numpy as np
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class KD_AnchorHeadSingle(AnchorHeadTemplate):
    '''
        This class will implement the distillation loss.
        This class do not support multihead.....
    '''
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.cls_stems = nn.Sequential(
            nn.Conv2d(
                input_channels, 256, 1, 1
            ),
            nn.Conv2d(
                256, 256, 3, 1, 1
            ),
            nn.Conv2d(
                256, 256, 3, 1, 1
            )
        )

        self.reg_stems = nn.Sequential(
            nn.Conv2d(
                input_channels, 256, 1, 1
            ),
            nn.Conv2d(
                256, 256, 3, 1, 1
            ),
            nn.Conv2d(
                256, 256, 3, 1, 1
            )
        )

        self.conv_cls = nn.Conv2d(
            256, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            256, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.conv_dir_cls = nn.Conv2d(
            256,
            self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
            kernel_size=1
        )
        sigma = torch.randn(2, requires_grad=True)
        self.sigma = nn.Parameter(sigma)
        self.knowledge_forward_rect = {}
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)


    def get_kd_reg_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.knowledge_forward_rect['box_reg_targets']
        box_cls_labels  = self.knowledge_forward_rect['box_cls_labels']

        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        anchors = torch.cat(self.anchors, dim=-3)

        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        teach_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            teach_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return teach_loss, tb_dict


    def get_kd_cls_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.knowledge_forward_rect['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def get_hint_loss(self):
        '''
            Reference the NeuroIPS 2020 Richness Feature Knowledge Distillation...
        '''
        #cost_function   = nn.MSELoss()
        cost_function   = nn.KLDivLoss(reduction='batchmean')
        student_feature = self.forward_ret_dict['student_feature']
        teacher_feature = self.knowledge_forward_rect['teacher_feature']
        teacher_mask    = self.knowledge_forward_rect['teacher_cls_preds']
        teacher_rich_feature = teacher_mask * teacher_feature
        fea_loss = cost_function(student_feature, teacher_feature)
        fea_loss = fea_loss / student_feature.shape[0]
        tb_dict = {
            'rpn_loss_feature': fea_loss.item()
        }
        return fea_loss, tb_dict



    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        kd_reg_loss, tb_dict_reg_teach = self.get_kd_reg_loss()
        kd_cls_loss, tb_dict_cls_teach = self.get_kd_cls_loss()
        kd_loss = kd_reg_loss + kd_cls_loss
        tb_dict.update(tb_dict_box)
        grund_rpn_loss = cls_loss + box_loss
        rpn_loss = kd_loss / (2 * self.sigma[0] ** 2) + grund_rpn_loss / (2 * self.sigma[1] ** 2) \
                   + torch.log(1 + self.sigma[0]**2) + torch.log(1 + self.sigma[1]**2)

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict



    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_temp  = self.cls_stems(spatial_features_2d)
        reg_temp  = self.reg_stems(spatial_features_2d)

        cls_preds = self.conv_cls(cls_temp)
        box_preds = self.conv_box(reg_temp)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
        self.forward_ret_dict['student_feature'] = spatial_features_2d
        self.knowledge_forward_rect['teacher_feature']  = data_dict['teacher_feature']
        self.knowledge_forward_rect['teacher_cls_pred'] = data_dict['teacher_cls_preds'] 

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(reg_temp)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            ### In here we should add some code to assign the target for soft label...
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

            teacher_dict = self.assign_targets(
                gt_boxes=data_dict['teacher_box']
            )
            self.knowledge_forward_rect.update(teacher_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
