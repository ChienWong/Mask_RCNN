import os
import sys
import random
import math
import numpy as np
import cv2
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
import tensorflow as tf
class PlaneConfig(Config):
    NAME="multiobject2"
    GPU_COUNT=1
    IMAGES_PER_GPU=2
    NUM_CLASSES=8
    IMAGE_MIN_DIM=1024
    IMAGE_MAX_DIM=1024
    RPN_ANCHOR_SCALES=(16,36,64,128,512)
    TRAIN_ROIS_PER_IMAGE=64
    STEPS_PER_EPOCH=100
    VALIDATION_STEPS=10
    LEARNING_RATE=0.002
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 0.
    }
    RPN_TRAIN_ANCHORS_PER_IMAGE=512
    TRAIN_ROIS_PER_IMAGE = 400
    RPN_NMS_THRESHOLD=0.8
    DETECTION_NMS_THRESHOLD=0.5
    MEAN_PIXEL = np.array([78.4, 82.7, 81.8]) 

class PlaneDataset(utils.Dataset):
    def loadDataset(self,count,path):
        self.add_class("object",1,"plane")
        self.add_class("object",2,"ship")
        self.add_class("object",3,"storage-tank")
        #self.add_class("object",4,"baseball-diamond")
        self.add_class("object",5,"tennis-court")
        #self.add_class("object",6,"basketball-court")
        #self.add_class("object",7,"ground-track-field")
        self.add_class("object",8,"harbor")
        #self.add_class("object",9,"bridge")
        self.add_class("object",10,"small-vehicle")
        self.add_class("object",11,"large-vehicle")
        #self.add_class("object",12,"round-about")
        #self.add_class("object",13,"swimming-pool")
        #self.add_class("object",14,"helicopter")
        #self.add_class("object",15,"soccer-ball-field")
        imgList=os.listdir(path+"/image")
        boxList=os.listdir(path+"/box")
        assert len(imgList)==len(boxList),"the count of image and box is not equality"
        assert len(imgList)>=count,"the count is larger than image"
        for i in range(count):
            img=cv2.imread(path+"/image/"+imgList[i])
            self.add_image("object",image_id=i,path=path+"/image/"+imgList[i],maskpath=path+"/box/"+boxList[i],width=img.shape[1],height=img.shape[0])

    def load_image(self,image_id):
        #print("image: "+str(image_id))
        info=self.image_info[image_id]
        path=info['path']
        return cv2.imread(path)

    def load_mask(self,image_id):
        #print("mask: "+str(image_id))
        info=self.image_info[image_id]
        maskpath=info['maskpath']
        file=open(maskpath)
        listmask=[]
        classid=[]
        for line in file:
            mask=np.zeros((self.image_info[image_id]["height"],self.image_info[image_id]["width"]),dtype=bool)
            box=line.split()
            if int(box[4])==8:
                continue
            x1=min(int(box[0]),int(box[2]))
            x2=max(int(box[0]),int(box[2]))
            y1=min(int(box[1]),int(box[3]))
            y2=max(int(box[1]),int(box[3]))
            if x2>mask.shape[1]:
                x2=mask.shape[1]
            if y2>mask.shape[0]:
                y2=mask.shape[0]
            for i in range(x1,x2-2):
                for j in range(y1,y2-2):
                    mask[j,i]=True
            listmask.append(mask)
            id=self.map_source_class_id("object."+box[4])
            classid.append(id)
        if len(classid) ==0:
            listmask.append(np.zeros((self.image_info[image_id]["height"],self.image_info[image_id]["width"]),dtype=bool))
            classid.append(0)
        listmask=np.stack(listmask,axis=2).astype(np.bool)
        classid=np.array(classid,dtype=np.int32)
        return listmask,classid

    def image_reference(self,image_id):
        info=self.image_info[image_id]
        if info["source"]=="object":
            return info["path"]
        else:
            super(PlaneDataset,self).image_reference(image_id)

dataset_train=PlaneDataset()
dataset_train.loadDataset(14348,"./multiobjectdataset/train")
dataset_train.prepare()
dataset_val=PlaneDataset()
dataset_val.loadDataset(4871,"./multiobjectdataset/val")
dataset_val.prepare()
config=PlaneConfig()
config.display()
model=modellib.MaskRCNN(mode="training",config=config,model_dir="./logs")
#model.load_weights("pretrained_weights.h5",by_name=True,exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])
model.load_weights(model.find_last(),by_name=True,exclude=["fpn_p1add","fpn_p2upsampled","fpn_c1p1","fpn_p1"])
model.train(dataset_train,dataset_val,learning_rate=config.LEARNING_RATE,epochs=100,layers="heads")
