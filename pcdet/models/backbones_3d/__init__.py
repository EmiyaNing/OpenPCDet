from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x, VoxelResBackDeepBone8x
from .spconv_unet import UNetV2
#from .votr_backbone import VoxelTransformerV3
from .spconv_multi_backbone import MultiVoxelBackBone8x
from .spconv_resnext import VoxelResXBackBone8x
from .IASSD_backbone import IASSD_Backbone
from .sa_spconv_backbone import SAVoxelBackBone8x, SAVoxelBackBone8xV2
from .spconv_big_kernal import VoxelNeXT

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    #'VoxelTransformerV3': VoxelTransformerV3,
    'VoxelResBackDeepBone8x': VoxelResBackDeepBone8x,
    'MultiVoxelBackBone8x': MultiVoxelBackBone8x,
    'VoxelResXBackBone8x': VoxelResXBackBone8x,
    'IASSD_Backbone': IASSD_Backbone,
    'SAVoxelBackBone8x': SAVoxelBackBone8x,
    'SAVoxelBackBone8xV2': SAVoxelBackBone8xV2,
    'VoxelNeXT': VoxelNeXT,
}
