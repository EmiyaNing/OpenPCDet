### The new version distillation recoder
1. Change the two-stage head's classification to three class
2. Using the distillation methods with tempture 2


### Now the result
bbox AP:90.8219, 90.0910, 89.2714
bev  AP:90.0486, 87.9664, 86.6708
3d   AP:89.0143, 78.6495, 77.2321
aos  AP:90.78, 89.94, 89.03
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.5742, 95.4161, 92.6592
bev  AP:93.1816, 89.2197, 88.3384
3d   AP:91.7039, 82.4229, 79.2927
aos  AP:96.51, 95.23, 92.38
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:74.6825, 69.4859, 67.5888
bev  AP:65.2713, 62.3445, 57.0509
3d   AP:62.3985, 56.5924, 54.3787
aos  AP:70.05, 64.04, 61.89
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:74.3177, 70.5194, 66.2580
bev  AP:65.7148, 60.9426, 55.7120
3d   AP:61.8297, 56.2347, 51.8368
aos  AP:69.27, 64.52, 60.15
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:87.2591, 80.4919, 74.5464
bev  AP:85.4563, 69.5059, 63.8796
3d   AP:82.0136, 66.2897, 60.9017
aos  AP:87.17, 79.97, 74.03
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:91.6199, 80.4817, 77.7501
bev  AP:89.2508, 69.7397, 65.4962
3d   AP:85.2744, 65.4603, 61.1632
aos  AP:91.51, 79.95, 77.09


### THe one-roi head self-distillation result recoder
1. Only use the voxel-rcnn head to teach the one-stage result
2. Using the dynamic weight to balance the regularize samples and try to get better result
2. Add a sub classification head to 3D backbone's after

### Now the result
2022-05-10 13:27:24,105   INFO  Generate label finished(sec_per_example: 0.0375 second).
2022-05-10 13:27:24,106   INFO  recall_roi_0.3: 0.000000
2022-05-10 13:27:24,106   INFO  recall_rcnn_0.3: 0.905239
2022-05-10 13:27:24,106   INFO  recall_roi_0.5: 0.000000
2022-05-10 13:27:24,106   INFO  recall_rcnn_0.5: 0.866913
2022-05-10 13:27:24,106   INFO  recall_roi_0.7: 0.000000
2022-05-10 13:27:24,106   INFO  recall_rcnn_0.7: 0.664920
2022-05-10 13:27:24,108   INFO  Average predicted number of objects(3769 samples): 6.114
2022-05-10 13:27:41,906   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7859, 89.9975, 89.1594
bev  AP:89.9452, 88.0403, 86.7859
3d   AP:88.3659, 78.4931, 77.1797
aos  AP:90.77, 89.88, 88.93
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.4668, 95.2875, 92.5997
bev  AP:93.1005, 91.0462, 88.4412
3d   AP:91.0761, 82.3310, 79.2745
aos  AP:96.43, 95.13, 92.33
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.9005, 69.1206, 67.2843
bev  AP:63.7829, 61.6777, 56.8584
3d   AP:60.3313, 55.7745, 53.6906
aos  AP:70.22, 65.40, 62.96
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:73.6454, 69.9746, 65.9067
bev  AP:62.9656, 60.2379, 55.5158
3d   AP:59.1558, 55.0886, 51.0911
aos  AP:69.45, 65.61, 61.24
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:87.4827, 80.4786, 74.1370
bev  AP:84.7037, 70.6187, 68.8938
3d   AP:81.5345, 67.5388, 61.8728
aos  AP:87.33, 79.43, 73.18
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:91.6069, 80.0769, 77.5795
bev  AP:88.4935, 72.0350, 67.9368
3d   AP:85.3545, 67.7162, 63.4950
aos  AP:91.40, 79.02, 76.38


### Obversion
1. This version self distillation could help detector get better accuracy on cyclist
2. But for Pedestrain the two-stage soft head could help the detector detect better
3. For outdoor 3D detection scernios the high tempture distillation could not help student get better accuracy(The number of class is too small)



### The voxel_self_second version 2
1. Keep the consistency of sub_classification and one-stage head's classification output
2. Using the sub heads as supplement classification heads.


### Now the dense head only best result
2022-05-11 05:36:21,739   INFO  Generate label finished(sec_per_example: 0.0354 second).
2022-05-11 05:36:21,739   INFO  recall_roi_0.3: 0.000000
2022-05-11 05:36:21,739   INFO  recall_rcnn_0.3: 0.907973
2022-05-11 05:36:21,739   INFO  recall_roi_0.5: 0.000000
2022-05-11 05:36:21,739   INFO  recall_rcnn_0.5: 0.869818
2022-05-11 05:36:21,739   INFO  recall_roi_0.7: 0.000000
2022-05-11 05:36:21,739   INFO  recall_rcnn_0.7: 0.668109
2022-05-11 05:36:21,742   INFO  Average predicted number of objects(3769 samples): 6.125
2022-05-11 05:36:39,592   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.8357, 90.0530, 89.1617
bev  AP:89.6506, 87.9082, 86.5583
3d   AP:88.5347, 78.4373, 77.1400
aos  AP:90.82, 89.92, 88.93
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.5376, 95.3792, 92.6346
bev  AP:92.7275, 90.8810, 88.1813
3d   AP:91.1840, 82.3592, 79.2073
aos  AP:96.51, 95.21, 92.36
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:74.8999, 69.7783, 67.5428
bev  AP:65.9417, 59.8010, 57.5381
3d   AP:62.3543, 56.5941, 51.2230
aos  AP:70.64, 65.12, 62.14
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:74.4918, 70.6459, 66.3161
bev  AP:66.3376, 60.7405, 56.4649
3d   AP:61.3205, 56.3273, 50.9942
aos  AP:69.87, 65.30, 60.62
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:87.4724, 75.7507, 74.7441
bev  AP:84.5432, 70.4978, 68.9945
3d   AP:82.5466, 67.8093, 62.6159
aos  AP:87.21, 74.07, 72.92
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:92.0089, 79.1775, 76.6109
bev  AP:88.5276, 70.7661, 67.7336
3d   AP:86.3299, 68.0322, 64.1065
aos  AP:91.69, 77.12, 74.54

### These version distillation framework add a complete sub head to 3D backbone
1. Add sub cls heads
2. Add sub box heads
3. Add sub dir heads
4. Using the gt_boxes to supervise the sub heads

### Now the results
##### One-stage head best result epoch 71
2022-05-13 11:04:02,549   INFO  Generate label finished(sec_per_example: 0.0372 second).
2022-05-13 11:04:02,549   INFO  recall_roi_0.3: 0.000000
2022-05-13 11:04:02,549   INFO  recall_rcnn_0.3: 0.906549
2022-05-13 11:04:02,549   INFO  recall_roi_0.5: 0.000000
2022-05-13 11:04:02,549   INFO  recall_rcnn_0.5: 0.867882
2022-05-13 11:04:02,550   INFO  recall_roi_0.7: 0.000000
2022-05-13 11:04:02,550   INFO  recall_rcnn_0.7: 0.663383
2022-05-13 11:04:02,552   INFO  Average predicted number of objects(3769 samples): 6.241
2022-05-13 11:04:22,637   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7922, 89.9470, 89.0533
bev  AP:89.8459, 87.6354, 86.4212
3d   AP:88.4478, 78.5266, 77.1208
aos  AP:90.78, 89.80, 88.81
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.3933, 94.9838, 92.4807
bev  AP:92.9748, 88.9800, 88.1258
3d   AP:89.6758, 82.2141, 79.2197
aos  AP:96.37, 94.79, 92.20
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.6280, 68.6525, 66.6293
bev  AP:64.6475, 62.1319, 56.9977
3d   AP:60.7829, 55.6364, 50.1598
aos  AP:70.28, 64.52, 61.89
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:73.4113, 69.4585, 65.3917
bev  AP:64.9406, 60.6635, 56.5041
3d   AP:59.5421, 55.1172, 49.8415
aos  AP:69.82, 64.77, 60.37
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:88.4713, 74.5665, 73.6362
bev  AP:86.3623, 70.6682, 65.3894
3d   AP:84.8526, 69.0884, 63.5597
aos  AP:88.17, 73.72, 72.69
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:93.4139, 75.4316, 72.8023
bev  AP:90.8821, 70.1690, 67.1158
3d   AP:89.1191, 68.1725, 63.9597
aos  AP:93.03, 74.53, 71.81

##### two-stage head best result epoch 80
2022-05-13 20:51:15,298   INFO  Generate label finished(sec_per_example: 0.0376 second).
2022-05-13 20:51:15,298   INFO  recall_roi_0.3: 0.972437
2022-05-13 20:51:15,298   INFO  recall_rcnn_0.3: 0.972380
2022-05-13 20:51:15,298   INFO  recall_roi_0.5: 0.934909
2022-05-13 20:51:15,298   INFO  recall_rcnn_0.5: 0.939408
2022-05-13 20:51:15,298   INFO  recall_roi_0.7: 0.719818
2022-05-13 20:51:15,298   INFO  recall_rcnn_0.7: 0.755923
2022-05-13 20:51:15,301   INFO  Average predicted number of objects(3769 samples): 9.451
2022-05-13 20:51:34,832   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:98.1680, 89.6075, 89.2376
bev  AP:90.1936, 88.2630, 87.7968
3d   AP:89.5476, 84.4517, 78.8578
aos  AP:98.13, 89.51, 89.08
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:99.0156, 94.5481, 94.2110
bev  AP:95.9328, 90.9993, 90.5139
3d   AP:92.8542, 85.1505, 82.5956
aos  AP:98.99, 94.42, 94.01
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.7716, 69.4689, 65.9306
bev  AP:68.2153, 61.8054, 58.3642
3d   AP:64.6562, 58.7669, 54.6135
aos  AP:71.38, 66.36, 62.65
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:74.5093, 69.5106, 66.1276
bev  AP:67.9658, 61.5986, 57.2405
3d   AP:65.0505, 58.0460, 53.5947
aos  AP:71.82, 66.06, 62.52
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:95.1753, 82.1425, 80.2394
bev  AP:93.6105, 75.4174, 73.5601
3d   AP:91.6348, 73.0464, 70.1992
aos  AP:94.89, 81.22, 79.24
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:96.6481, 84.1023, 81.2205
bev  AP:95.2897, 78.0818, 74.8215
3d   AP:92.9171, 74.0948, 70.8252
aos  AP:96.37, 83.09, 80.20




### result obversion
1. the distillation loss will decrease in the training process(Very good, its the first time)
2. the Cyclist's result is very close to its two-stage heads
3. And obverse the two-stage heads the result also get improvement.

### The decouple distillation version self_distillation
1. Reference the "Decouple Distillation", we use the two-stage results to distill the target class only



### Now the result

##### One-stage head's output
2022-05-16 03:10:44,518   INFO  Generate label finished(sec_per_example: 0.0361 second).
2022-05-16 03:10:44,518   INFO  recall_roi_0.3: 0.000000
2022-05-16 03:10:44,518   INFO  recall_rcnn_0.3: 0.904784
2022-05-16 03:10:44,518   INFO  recall_roi_0.5: 0.000000
2022-05-16 03:10:44,518   INFO  recall_rcnn_0.5: 0.866230
2022-05-16 03:10:44,518   INFO  recall_roi_0.7: 0.000000
2022-05-16 03:10:44,518   INFO  recall_rcnn_0.7: 0.665148
2022-05-16 03:10:44,521   INFO  Average predicted number of objects(3769 samples): 6.084
2022-05-16 03:11:04,515   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7786, 90.0536, 89.2370
bev  AP:89.9355, 88.0103, 86.7933
3d   AP:88.6202, 78.3846, 77.1480
aos  AP:90.75, 89.92, 89.02
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.4791, 95.3357, 92.6525
bev  AP:93.0320, 91.0227, 88.4317
3d   AP:91.2093, 82.2115, 79.2402
aos  AP:96.44, 95.16, 92.40
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:74.9027, 70.0407, 67.9434
bev  AP:64.0341, 61.6200, 56.0383
3d   AP:60.4090, 55.4609, 49.7990
aos  AP:71.19, 65.62, 63.27
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:74.9469, 72.2834, 68.0332
bev  AP:64.3550, 60.2039, 54.8450
3d   AP:60.0001, 54.9793, 49.6156
aos  AP:70.99, 67.33, 62.89
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:88.1802, 77.1058, 75.4569
bev  AP:85.8558, 72.0858, 65.8403
3d   AP:84.5455, 70.1066, 64.1122
aos  AP:88.02, 76.58, 74.79
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:93.0722, 80.5436, 76.1081
bev  AP:90.1331, 72.3543, 67.9555
3d   AP:88.4519, 69.2534, 64.8871
aos  AP:92.84, 79.89, 75.39

##### Two-stage head's output
2022-05-16 11:08:50,163   INFO  Generate label finished(sec_per_example: 0.0374 second).
2022-05-16 11:08:50,163   INFO  recall_roi_0.3: 0.972494
2022-05-16 11:08:50,163   INFO  recall_rcnn_0.3: 0.972779
2022-05-16 11:08:50,163   INFO  recall_roi_0.5: 0.935535
2022-05-16 11:08:50,163   INFO  recall_rcnn_0.5: 0.939294
2022-05-16 11:08:50,163   INFO  recall_roi_0.7: 0.721469
2022-05-16 11:08:50,163   INFO  recall_rcnn_0.7: 0.761048
2022-05-16 11:08:50,165   INFO  Average predicted number of objects(3769 samples): 7.685
2022-05-16 11:09:10,064   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:97.6334, 89.5119, 89.2240
bev  AP:90.2912, 88.3600, 87.8640
3d   AP:89.3637, 84.4952, 78.8787
aos  AP:97.58, 89.39, 89.05
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.6726, 94.7946, 94.3920
bev  AP:95.8808, 91.3022, 89.0810
3d   AP:92.5756, 85.2178, 82.8900
aos  AP:98.63, 94.63, 94.16
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:75.8197, 72.4484, 68.0554
bev  AP:69.8571, 63.7693, 60.7982
3d   AP:65.7754, 60.7199, 55.9160
aos  AP:72.86, 68.89, 64.25
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:76.7025, 72.6547, 69.6628
bev  AP:70.1712, 63.9398, 59.3061
3d   AP:66.9380, 59.9752, 55.1495
aos  AP:73.48, 68.82, 65.44
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:96.0968, 83.3192, 80.4357
bev  AP:93.6945, 74.6701, 72.0197
3d   AP:87.1092, 73.0968, 68.9770
aos  AP:95.90, 82.71, 79.77
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:97.1412, 84.5988, 81.3215
bev  AP:94.6510, 77.0409, 72.4287
3d   AP:91.9156, 73.9043, 69.3446
aos  AP:96.95, 84.01, 80.62



### Obversion and analysis
1. the cyclist get  very godd performance.
2. The performance of Pedestrain get a little decrease.
3. The performance of Car 3D keep unchange, but the performance of bev get improvement.