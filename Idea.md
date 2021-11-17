### First stage idea
1. Use the Second as the baseline
2. Add a transformer layer in Bevbackbone to extend the receptive field
3. Decouple the class head, box head, and iou head
4. Add a simOTA target assign for this model
5. Add a sub-segment branch to cos SA-SSD
6. Multi positive used in yolox and fcos
7. Add a BEV feature input branch for BEVRPNbackbone
8. Add a Deeplab style dilate conv 3d block to extend the receptive field
9. Add a coordinate 3d attention for this model

### Hard rank
1 *
2 **
3 ***
4 ****
5 ***
6 ****
7 **
8 **
9 ***

### Implement order
1 -> 2 -> 5 -> 7 -> 9 -> 3 -> 6 -> 4 -> 8
3 is the task's pre_task

### 2021/11/14 Works
1. Finish the cpp_extension's build for OpenPCDet, and attuned for C++ CUDA program
2. Train the Second and VoxelRCNN
3. Attuned the OpenPCDet's anchor_generator's data flow
4. Attuned the OpenPCDet's detector3d_template.py data flow