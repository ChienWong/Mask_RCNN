## mask rcnn具体原理见[Matterport Mask RCNN](https://github.com/matterport/Mask_RCNN)
# 具体修改部分
### 增加C1特征层至mrcnn分支中，以应对小物体如车辆的检测
### 修改mrcnn损失函数为focal loss损失,解决训练样本分布不均衡问题
### 修改训练数据生成部分代码，使用keras.utils.Sequence代替python迭代器，更安全，避免加载相同数据
# 效果
### ![汽车](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/61.jpg)![汽车](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/3664.jpg)
### ![飞机](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/438.jpg)
### ![港口](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/1841.jpg)
### ![网球场](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/474.jpg)
# 数据集（DOTA-v1.5)
### 下载地址(https://captain-whu.github.io/DOAI2019/dataset.html)
### 请自行转换格式
# 使用该项目
### 请在train.py中配置相关参数，如何配置请参考(https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/train_shapes.ipynb)
### 请按格式配置数据集，参考multiobjectdataset下的文件
### 运行train.py
## focal loss使用
### focal loss下降缓慢，建议先使用交叉熵损失，后使用focal loss
### 默认使用交叉熵损失，若要使用focal loss,请取消注释model.py的1117行并修改对应的alpha值，取消注释1121行，注释1114行
