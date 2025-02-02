from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector, build_fusion_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def build_fusion_network(model1, model2 ,model_cfg1, num_class, dataset):
    model = build_fusion_detector(model1, model2, model_cfg1, num_class, dataset)
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def load_data_to_gpu_tta(batch_dict_list):
    for batch_dict in batch_dict_list:
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib']:
                continue
            elif key in ['images']:
                batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = torch.from_numpy(val).int().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


def teacher_model_fn_decorator():
    ModelReturn = namedtuple('TeacherReturn', ['predict_dict', 'recall_dict', 'teacher_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        predicts, recall_dict, teacher_dict = model(batch_dict)

        return ModelReturn(predicts, recall_dict, teacher_dict)

    return model_func




def ema_model_fn_decorator(model, batch_dict):
    load_data_to_gpu(batch_dict)
    ema_dict = model(batch_dict)

    return ema_dict
