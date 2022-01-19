### Add Knowledge distillation to pcdet
1. use the latest version of pcdet and latest spconv
2. build a new distillation train script for OpenPCDet
3. try to construct a two stage model online teach the one stage model
4. try to use the ODiou loss to train

### Work date
1. implement ODiou loss to OpenPCdet
> 1.1 try to understand overall sence of axis_aligned_target_assigner
> 1.2 try to catch write some code to caculate each box's iou value with gt_box

2. try to combine the teacher student model to our works(Voxel RCNN && SPVB-SSD).
> 2.1 try to combine voxel-RCNN and votr
> 2.2 try to train a best votr-voxel-RCNN
> 2.3 try to use votr-voxel-RCNN as teacher to generage soft-label
> 2.4 try to construct a consistency loss between teacher and student
> 2.5 try to get a knowledge-distillation model

