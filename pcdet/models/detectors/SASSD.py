import torch
import torch.nn as nn
from numpy.lib.function_base import disp
from .detector3d_template import Detector3DTemplate
from ...utils import common_utils
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.loss_utils import SigmoidFocalClassificationLoss, WeightedSmoothL1Loss
from ..model_utils.meter_utils import AverageMeter

class SASSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.voxel_size  = [0.05, 0.05, 0.1]
        self.point_cloud_range = [0, -40, -3, 70.4, 40, 1]
        self.forward_ret_dict = {}
        self.aux_cls_costfucntion = SigmoidFocalClassificationLoss()
        self.aux_ctr_costfunction = WeightedSmoothL1Loss(code_weights=None)
        self.point_mlp   = nn.Linear(160, 64, bias=False)
        self.point_cls   = nn.Linear(64, 1, bias=False)
        self.point_reg   = nn.Linear(64, 3, bias=False)
        self.rpn_loss_avg = AverageMeter()
        self.aux_cls_avg  = AverageMeter()
        self.aux_reg_avg  = AverageMeter()


    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        self.forward_ret_dict['gt_boxes']   = batch_dict['gt_boxes']
        self.forward_ret_dict['batch_size'] = batch_dict['batch_size']
        self.forward_ret_dict['x_conv2'] = batch_dict['multi_scale_3d_features']['x_conv2']
        self.forward_ret_dict['x_conv3'] = batch_dict['multi_scale_3d_features']['x_conv3']
        self.forward_ret_dict['x_conv4'] = batch_dict['multi_scale_3d_features']['x_conv4']
        self.forward_ret_dict['point']   = batch_dict['points']

        if self.training:
            point_features = self.inteplote_sparse_features()
            point_temp     = self.point_mlp(point_features)
            self.forward_ret_dict['point_cls'] = self.point_cls(point_temp)
            self.forward_ret_dict['point_reg'] = self.point_reg(point_temp)
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts


    def inteplote_sparse_features(self):
        batch_size = self.forward_ret_dict['batch_size']
        point   = self.forward_ret_dict['point']
        x_conv2 = self.forward_ret_dict['x_conv2']
        x_conv3 = self.forward_ret_dict['x_conv3']
        x_conv4 = self.forward_ret_dict['x_conv4']
        conv2_feat, conv2_coor = x_conv2.features, x_conv2.indices
        conv3_feat, conv3_coor = x_conv3.features, x_conv3.indices
        conv4_feat, conv4_coor = x_conv4.features, x_conv4.indices
        conv2_bs_idx, conv3_bs_idx, conv4_bs_idx = conv2_coor[:, 0], conv3_coor[:, 0], conv4_coor[:, 0]
        point_bs_idx = point[:, 0]
        point_xyz    = point[:, 1:]
        center_conv2 = common_utils.get_voxel_centers(
            voxel_coords=conv2_coor[:, 1:],
            downsample_times=2,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        point_feat2  = self.inteplote_one_stride_features(batch_size, conv2_bs_idx, point_bs_idx, center_conv2, point_xyz, conv2_feat)

        center_conv3 = common_utils.get_voxel_centers(
            voxel_coords=conv3_coor[:, 1:],
            downsample_times=4,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        point_feat3  = self.inteplote_one_stride_features(batch_size, conv3_bs_idx, point_bs_idx, center_conv3, point_xyz, conv3_feat)

        center_conv4 = common_utils.get_voxel_centers(
            voxel_coords=conv4_coor[:, 1:],
            downsample_times=8,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        point_feat4  = self.inteplote_one_stride_features(batch_size, conv4_bs_idx, point_bs_idx, center_conv4, point_xyz, conv4_feat)

        point_feature = torch.cat([point_feat2, point_feat3, point_feat4], dim=1)
        return point_feature
        


    def inteplote_one_stride_features(self, batch_size, voxel_bs_idx, point_bs_idx, voxel_center, point, features):
        _, C = features.shape
        point_features = torch.zeros([point.shape[0], C]).float().cuda()
        for bs_idx in range(batch_size):
            voxel_bs_mask = (voxel_bs_idx == bs_idx)
            point_bs_mask = (point_bs_idx == bs_idx)

            single_stride_feats = pointnet2_utils.top3_interpolate(
                xyz=voxel_center[voxel_bs_mask],
                new_xyz=point[point_bs_mask][:, 1:].contiguous(),
                feats=features[voxel_bs_mask],
            )
            point_features[point_bs_mask] = single_stride_feats
        
        return point_features
        
    def build_aux_target(self, batch_size, point, enlarge=1.0):
        center_offset = torch.zeros([point.shape[0], 3]).float().cuda()
        #point_labels  = torch.zeros([point.shape[0]]).long().cuda()
        point_label_list = []
        center_label_list= []
        for i in range(batch_size):
            batch_one_gt_boxes = self.forward_ret_dict['gt_boxes'][i]
            k = batch_one_gt_boxes.__len__() - 1
            while k > 0 and batch_one_gt_boxes[k].sum() == 0:
                k -= 1
            batch_one_gt_boxes = batch_one_gt_boxes[:k+1]
            batch_one_gt_boxes[:, 3:6] *= enlarge
            batch_one_pt_mask  = point[:, 0] == i
            batch_one_point    = point[batch_one_pt_mask][:, 1:4]
            #one_masks          = torch.ones(batch_one_pt_mask.sum()).long().cuda()
            
            box_idx_of_pts     = roiaware_pool3d_utils.points_in_boxes_gpu(batch_one_point.unsqueeze(0), batch_one_gt_boxes.unsqueeze(0)[:, :, :7].contiguous()).long().squeeze(dim=0)
            fg_flag            = (box_idx_of_pts >= 0)
            center_temp_labels = torch.zeros([fg_flag.shape[0], 3]).float().cuda()
            gt_boxes_of_points = batch_one_gt_boxes[box_idx_of_pts[fg_flag]]
            center_temp_labels[fg_flag] = gt_boxes_of_points[:, :3].float()
            point_label_list.append(fg_flag.long())
            center_label_list.append(center_temp_labels)

        point_labels = torch.cat(point_label_list, dim=0)
        center_offset= torch.cat(center_label_list, dim=0)
        
        return center_offset, point_labels

    def aux_loss(self):
        batch_size = self.forward_ret_dict['batch_size']
        points     = self.forward_ret_dict['point']
        point_cls  = self.forward_ret_dict['point_cls']
        point_reg  = self.forward_ret_dict['point_reg']
        center_labels, pts_labels = self.build_aux_target(batch_size, points)


        cls_targets = pts_labels.float()
        cls_targets = cls_targets.unsqueeze(-1)
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights    = pos + neg
        cls_weights    = cls_weights / pos_normalizer

        reg_weights    = pos
        reg_weights    = reg_weights / pos_normalizer

        aux_loss_cls   = self.aux_cls_costfucntion(point_cls.unsqueeze(0), cls_targets.unsqueeze(0), cls_weights.unsqueeze(0)).sum()
        aux_loss_cls   /= batch_size

        aux_loss_reg   = self.aux_ctr_costfunction(point_reg.unsqueeze(0), center_labels.unsqueeze(0), reg_weights.unsqueeze(0)).sum()
        aux_loss_reg   /= batch_size
        self.aux_cls_avg.update(aux_loss_cls.item())
        self.aux_reg_avg.update(aux_loss_reg.item())
        tb_dict = {
            'aux_loss_cls': aux_loss_cls.item(),
            'aux_loss_reg': aux_loss_reg.item(),
            'mean_aux_loss_cls': self.aux_cls_avg.avg,
            'mean_aux_loss_reg': self.aux_reg_avg.avg,
        }
        return aux_loss_cls, aux_loss_reg, tb_dict



    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        aux_cls_loss, aux_reg_loss, tb_aux_dict = self.aux_loss()
        self.rpn_loss_avg.update(loss_rpn.item())

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'mean_loss_rpn': self.rpn_loss_avg.avg,
            **tb_dict
        }
        tb_dict.update(tb_aux_dict)

        loss = loss_rpn + aux_cls_loss * 0.1 + aux_reg_loss * 0.1
        loss = loss_rpn
        return loss, tb_dict, disp_dict
