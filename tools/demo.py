import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
import cv2

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def removePoints(self, PointCloud, BoundaryCond):
        # Boundary condition
        minX = BoundaryCond['minX']
        maxX = BoundaryCond['maxX']
        minY = BoundaryCond['minY']
        maxY = BoundaryCond['maxY']
        minZ = BoundaryCond['minZ']
        maxZ = BoundaryCond['maxZ']

        # Remove the point out of range x,y,z
        mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
                PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
        PointCloud = PointCloud[mask]

        PointCloud[:, 2] = PointCloud[:, 2] - minZ

        return PointCloud

    def makeBEVFeature(self, points, boundry, Map_config):
        points = self.removePoints(points, boundry)
        discretization_x = Map_config["discretization_x"]
        discretization_y = Map_config["discretization_y"]
        Width   = Map_config["BEVHeight"]
        Height  = Map_config["BEVWidth"]

        PointCloud = np.copy(points)
        PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / discretization_x) + Height / 2)
        PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / discretization_y))
        indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
        PointCloud = PointCloud[indices]

        heightMap  = np.zeros((Height, Width))
        _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
        PointCloud_frac = PointCloud[indices]
        # some important problem is image coordinate is (y,x), not (x,y)
        # so here just use x as height, y as width
        max_height = float(np.abs(boundry['maxZ'] - boundry['minZ']))
        heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

        # Intensity Map & DensityMap
        intensityMap = np.zeros((Height, Width))
        densityMap = np.zeros((Height, Width))

        _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
        PointCloud_top = PointCloud[indices]

        normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

        intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
        densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

        RGB_Map = np.zeros((3, Height, Width))
        RGB_Map[2, :, :] = densityMap[:Height, :Width]  # r_map
        RGB_Map[1, :, :] = heightMap[:Height, :Width]  # g_map
        RGB_Map[0, :, :] = intensityMap[:Height, :Width]  # b_map
        RGB_Map = np.transpose(RGB_Map, [0, 2, 1])
        return RGB_Map


    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        boundry = {}
        point_range = self.dataset_cfg.POINT_CLOUD_RANGE
        voxel_step  = self.dataset_cfg.DATA_PROCESSOR[2].VOXEL_SIZE
        MAP_config  = self.dataset_cfg.BEV_CONFIG
        boundry["minX"] = point_range[1]
        boundry["minY"] = point_range[0]
        boundry["minZ"] = point_range[2]
        boundry["maxX"] = point_range[4]
        boundry["maxY"] = point_range[3]
        boundry["maxZ"] = point_range[5]
        boundry["step_x"] = voxel_step[1]
        boundry["step_y"] = voxel_step[0]
        #print("Start to transfor bev image")
        rgb_map = self.makeBEVFeature(points, boundry, MAP_config)
        #print("BEV image transfor finish")
        print_map = np.floor(rgb_map * 255)
        print_map = print_map.astype(np.uint8)
        print_map = print_map.transpose(1, 2, 0)
        cv2.imshow("bev_image", print_map)



        input_dict = {
            'points': points,
            'frame_id': index,
            'bev': rgb_map
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            bev_feature   = pred_dicts[0]['bev_feature']
            print(bev_feature.shape)
            scores = pred_dicts[0]['pred_scores']
            boxes  = pred_dicts[0]['pred_boxes']
            labels = pred_dicts[0]['pred_labels']
            indices = scores > 0.5
            filter_box = boxes[indices]
            filter_sco = scores[indices]
            filter_lab = labels[indices]
        
            


            '''V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )'''
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=filter_box,
                ref_scores=filter_sco, ref_labels=filter_lab
            )
            mlab.show(stop=True)
            

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
