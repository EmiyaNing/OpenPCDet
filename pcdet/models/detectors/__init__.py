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
from .voxel_secondv3 import Voxel_Second_Subnet
from .voxel_sub_second import Voxel_Second_Self
from .iou_detector.second_iou import IOU_SECONDNet
from .self_sub_heads.voxel_subheadsv1 import Voxel_SubHeads
from .self_sub_heads.voxel_subheadsv2 import Voxel_SubHeadsv2
from .self_sub_heads.voxel_subheadsv3 import Voxel_DESubHeads
from .self_sub_heads.voxel_subheadsv4 import Voxel_PASubHeads
from .self_sub_heads.voxel_subhead_se import Voxel_SESubHeads
from .self_sub_heads.voxel_subhead_simple import Voxel_BTSubHeads
from .self_sub_heads.voxel_subhead_simplev2 import Voxel_BTSubHeadsv2
from .self_sub_heads.voxel_subhead_batch import Voxel_PerSubHeadsv2
from .self_sub_heads.voxel_subhead_batchv2 import Voxel_PerSubHeadsv3
from .self_sub_heads.voxel_subhead_batchv3 import Voxel_PerSubHeadsv4
from .self_sub_heads.voxel_subhead_target import Voxel_TASubHeads
from .self_compent.voxel_debeta import Voxel_DEBETA
from .self_compent.voxel_verfy_debeta import Voxel_VDEBETA
from .self_compent.voxel_simple_f import Voxel_SubHeadsF
from .self_compent.voxel_simple import Voxel_Simple
from .self_compent.voxel_siou import Voxel_Siou
from .self_compent.voxel_sfeat import Voxel_Feat
from .self_compent.voxel_sort import Voxel_Sort
from .sienet import SIENet

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
    'Voxel_Second_Subnet': Voxel_Second_Subnet,
    'Voxel_Second_Self': Voxel_Second_Self,
    'IOU_SECONDNet': IOU_SECONDNet,
    'Voxel_SubHeads': Voxel_SubHeads,
    'Voxel_SubHeadsv2': Voxel_SubHeadsv2,
    'Voxel_DESubHeads': Voxel_DESubHeads,
    'SIENet': SIENet,
    'Voxel_PASubHeads': Voxel_PASubHeads,
    'Voxel_SESubHeads': Voxel_SESubHeads,
    'Voxel_BTSubHeads': Voxel_BTSubHeads,
    'Voxel_BTSubHeadsv2': Voxel_BTSubHeadsv2,
    'Voxel_PerSubHeadsv2': Voxel_PerSubHeadsv2,
    'Voxel_PerSubHeadsv3': Voxel_PerSubHeadsv3,
    'Voxel_PerSubHeadsv4': Voxel_PerSubHeadsv4,
    'Voxel_TASubHeads': Voxel_TASubHeads,
    'Voxel_DEBETA': Voxel_DEBETA,
    'Voxel_VDEBETA': Voxel_VDEBETA,
    'Voxel_SubHeadsF': Voxel_SubHeadsF,
    'Voxel_Simple': Voxel_Simple,
    'Voxel_Siou': Voxel_Siou,
    'Voxel_Feat': Voxel_Feat,
    'Voxel_Sort': Voxel_Sort,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model


def build_fusion_detector(model1, model2, model_cfg1, num_class, dataset):

    fusion_model = FUSIONNet(model_cfg=model_cfg1, num_class=num_class, dataset=dataset, model_1=model1, model_2=model2)

    return fusion_model