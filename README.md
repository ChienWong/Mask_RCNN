## mask rcnn具体原理见[Matterport Mask RCNN](https://github.com/matterport/Mask_RCNN)
# 具体修改部分
### 增加C1特征层至mrcnn分支中，以应对小物体如车辆的检测
### 修改mrcnn损失函数为focal loss损失,解决训练样本分布不均衡问题
### 修改训练数据生成部分代码，使用keras.utils.Sequence代替python迭代器，更安全，避免加载相同数据
