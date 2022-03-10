from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN, PVRCNN_UN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN, VoxelRCNN_UN
from .pv_second import PVSECONDNet
from .centerpoint import CenterPoint
from .votr_tsd_net import VoTrRCNN
from .se_second import SESSD

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'PVSECONDNet': PVSECONDNet,
    'CenterPoint': CenterPoint,
    'PVRCNN_UN': PVRCNN_UN,
    'VoxelRCNN_UN': VoxelRCNN_UN,
    'VoTrRCNN': VoTrRCNN,
    'SESSD': SESSD,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
