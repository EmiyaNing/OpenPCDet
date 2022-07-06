from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate
from .ct3d_head import CT3DHead
from .voxelrcnn_new import VoxelRCNNHead_New
from .sie_head import SIEHead
from .pdv_head import PDVHead


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'SECONDHead': SECONDHead,
    'PointRCNNHead': PointRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'CT3DHead': CT3DHead,
    'VoxelRCNNHead_New': VoxelRCNNHead_New,
    'SIEHead': SIEHead,
    'PDVHead': PDVHead,
}
