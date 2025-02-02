import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, object3d_kitti, calibration_kitti, box_utils

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
        self.info = {}
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def get_label(self, idx):
        if idx < 10:
            label_file = self.root_path.parent / 'label_2' / ('00000%s.txt' % idx)
        elif idx < 100:
            label_file = self.root_path.parent / 'label_2' / ('0000%s.txt' % idx)
        else:
            label_file = self.root_path.parent / 'label_2' / ('000%s.txt' % idx)
        print(label_file)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)


    def get_calib(self, idx):
        if idx < 10:
            calib_file = self.root_path.parent / 'calib' / ('00000%s.txt' % idx)
        elif idx < 100:
            calib_file = self.root_path.parent / 'calib' / ('0000%s.txt' % idx)
        else:
            calib_file = self.root_path.parent / 'calib' / ('000%s.txt' % idx)
        print(calib_file)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_info(self, idx):
        obj_list = self.get_label(idx)
        calib = self.get_calib(idx)
        info = {}
        annotations = {}
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
        annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
        annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
        annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        annotations['score'] = np.array([obj.score for obj in obj_list])
        annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
        num_gt = len(annotations['name'])
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)

        loc = annotations['location'][:num_objects]
        dims = annotations['dimensions'][:num_objects]
        rots = annotations['rotation_y'][:num_objects]
        loc_lidar = calib.rect_to_lidar(loc)
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
        annotations['gt_boxes_lidar'] = gt_boxes_lidar
        info['annos'] = annotations
        return info

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        calib = self.get_calib(index)
        info = self.get_info(index)
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })


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
            gt_boxes = data_dict['gt_boxes'][0, :,:7]
            print(gt_boxes)
            if gt_boxes.shape[1] > 0:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], gt_boxes=gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )
            else:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()