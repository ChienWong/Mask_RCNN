## mask rcnn for details see [Matterport Mask RCNN](https://github.com/matterport/Mask_RCNN)
# Specific modification
### add C1 feature map to mrcnn branch,for detecting the small object such as cars
### modify the loss of classification in mrchh branch to focal loss,for blancing the number of different object
### modify the code of generator of data, using keras.utils.Sequence replacing it, more safe and avoiding loading same image
# Result
### ![car](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/61.jpg)![汽车](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/3664.jpg)
### ![plane](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/438.jpg)
### ![harbor](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/1841.jpg)
### ![tennis](https://github.com/mohuazheliu/Mask_RCNN/blob/master/images/474.jpg)
# Dataset(DOTA-v1.5)
### [Download](https://captain-whu.github.io/DOAI2019/dataset.html)
### Please to tranform the format
# How to run it
### Configure related parameters in train.py, refer to [here](https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/train_shapes.ipynb)
### Configure the dataset by format,refer to the readme in multiobjectdataset
### run train.py
## Using focal loss
### Slow decline of focal loss,It is recommended to use cross entropy loss first, and then focal loss
### Using cross entropy loss by deflaut,if you want to use it,uncomment line 1117 of model.py and modify the corresponding alpha value, uncomment line 1121 and uncomment line 1114
