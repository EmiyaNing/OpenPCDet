from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN, PVRCNN_UN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN, VoxelRCNN_TTA
from .pv_second import PVSECONDNet
from .centerpoint import CenterPoint
from .votr_tsd_net import VoTrRCNN
from .se_second import SESSD
from .voxel_teacher import VoxelTEACHRCNN, VoxelTEACHRCNNV2
from .fusion_model import FUSIONNet
from .pv_rcnn_teacher import PVRCNN_Teacher
from .Multi_centerpoint import MultiCenterPoint
from .voxel_second import VoxelSECOND
from .semi_second import SemiSECOND
from .voxel_secondv2 import Self_VoxelSECOND
from .CT3D import CT3D
from .CT3D_3CAT import CT3D_3CAT
from .voxel_ct3d import Voxel_CT3D
from .voxel_ct3d_uncertaity import Voxel_CT3D_Uncertainty
from .IASSD import IASSD
from .second_view import SECONDNet_view
from .second_test import FromVoxel2Point
from .self_voxel_scconv import Voxel_SCCONV
from .SASSD import SASSD
from .voxel_ct3dv2 import Voxel_CT3DV2

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
    'VoxelRCNN_TTA': VoxelRCNN_TTA,
    'VoTrRCNN': VoTrRCNN,
    'SESSD': SESSD,
    'VoxelTEACHRCNN': VoxelTEACHRCNN,
    'VoxelTEACHRCNNV2': VoxelTEACHRCNNV2,
    'PVRCNN_Teacher': PVRCNN_Teacher,
    'MultiCenterPoint': MultiCenterPoint,
    'VoxelSECOND': VoxelSECOND,
    'SemiSECOND': SemiSECOND,
    'Self_VoxelSECOND': Self_VoxelSECOND,
    'CT3D': CT3D,
    'CT3D_3CAT': CT3D_3CAT,
    'Voxel_CT3D': Voxel_CT3D,
    'Voxel_CT3D_Uncertainty': Voxel_CT3D_Uncertainty,
    'IASSD': IASSD,
    'SECONDNet_view': SECONDNet_view,
    'FromVoxel2Point': FromVoxel2Point,
    'Voxel_SCCONV': Voxel_SCCONV,
    'SASSD': SASSD,
    'Voxel_CT3DV2': Voxel_CT3DV2,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model


def build_fusion_detector(model1, model2, model_cfg1, num_class, dataset):

    fusion_model = FUSIONNet(model_cfg=model_cfg1, num_class=num_class, dataset=dataset, model_1=model1, model_2=model2)

    return fusion_model