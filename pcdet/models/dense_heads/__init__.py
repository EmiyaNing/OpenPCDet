from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle, DecoupleAnchorHeadSingle,DecoupleHeadThree, DecoupleHeadFour, DecoupleHeadThreeLightweight
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .distillation_anchor_head import KD_AnchorHeadSingle
from .GID_anchor_head import GID_AnchorHeadSingle
from .GID_anchor_headv2 import SESSD_Head
from .iou_head import ODiou_Head
from .diversity_distillation_head import Diversity_Head
from .diversity_distillation_headv2 import Diversity_HeadV2
from .diversity_distillation_headv3 import Diversity_HeadV3
from .IASSD_head import IASSD_Head
from .anchor_head_harmony import AnchorHeadHarmony

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'DecoupleAnchorHeadSingle': DecoupleAnchorHeadSingle,
    'DecoupleHeadThree': DecoupleHeadThree,
    'DecoupleHeadFour': DecoupleHeadFour,
    'CenterHead': CenterHead,
    'DecoupleHeadThreeLightweight': DecoupleHeadThreeLightweight,
    'KD_AnchorHeadSingle': KD_AnchorHeadSingle,
    'GID_AnchorHeadSingle': GID_AnchorHeadSingle,
    'SESSD_Head': SESSD_Head,
    'ODiou_Head': ODiou_Head,
    'Diversity_Head': Diversity_Head,
    'Diversity_HeadV2': Diversity_HeadV2,
    'Diversity_HeadV3': Diversity_HeadV3,
    'IASSD_Head': IASSD_Head,
    'AnchorHeadHarmony': AnchorHeadHarmony
}
