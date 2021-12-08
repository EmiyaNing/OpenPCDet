from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle, DecoupleAnchorHeadSingle,DecoupleHeadThree, DecoupleHeadFour
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead

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
}
