from urllib import request
import re
import urllib.request
import os
import random
import math
import numpy as np
import cv2
import time
agents = [

'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)Chrome/60.0.3112.101Safari/537.36',

'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/532.5 (KHTML, like Gecko) Chrome/4.0.249.0 Safari/532.5',

'Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/532.9 (KHTML, like Gecko) Chrome/5.0.310.0 Safari/532.9',

'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/534.7 (KHTML, like Gecko) Chrome/7.0.514.0 Safari/534.7',

'Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US) AppleWebKit/534.14 (KHTML, like Gecko) Chrome/9.0.601.0 Safari/534.14',

'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.14 (KHTML, like Gecko) Chrome/10.0.601.0 Safari/534.14',
'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.20 (KHTML, like Gecko) Chrome/11.0.672.2 Safari/534.20',
'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.27 (KHTML, like Gecko) Chrome/12.0.712.0 Safari/534.27',

'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.1 (KHTML, like Gecko) Chrome/13.0.782.24 Safari/535.1']


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def getimg(Tpath, x, y):
    try:
        req = urllib.request.Request(Tpath)
        req.add_header('User-Agent', random.choice(agents))
        pic = urllib.request.urlopen(req, timeout=60)
        image = np.asarray(bytearray(pic.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print(str(x) + '_' + str(y) + 'success')
        return image
    except Exception as e:
        print(e)
        print(str(x) + '_' + str(y) + 'fail,try again')
        time.sleep(10)
        getimg(Tpath, x, y)

path = r"./yancheng-map"
zoom = 18
left,top = deg2num(33.42,120.05, zoom)  
right,bottom = deg2num(33.28,120.27, zoom)

imgdir=os.listdir(path);
if len(imgdir)!=0:
    last=imgdir[-1]
    last=last.replace(".png","")
    last=last.split("_")
    left=int(last[0])
    top=int(last[1])+4

print(str(left))
print(str(right))
print(str(top))
print(str(bottom))
print(str(left - right))
print(str(top - bottom))

for x in range(left, right,4):
    print(x)
    for y in range(top, bottom,4):
        image=np.empty(shape=(0,1024,3),dtype="uint8")
        for i in range(4):
            img=np.empty(shape=(256,0,3),dtype="uint8")
            for j in range(4):
                tilepath="http://www.google.cn/maps/vt?lyrs=s@815&gl=cn&x="+str(x+j)+"&y="+str(y+i)+"&z="+str(zoom)
                m=getimg(tilepath, x+j, y+i)
                img=np.concatenate((img,m),axis=1)
            image=np.concatenate((image,img),axis=0)
        #path=os.path.join(path, str(x) + "_" + str(y) + ".png"),
        cv2.imwrite(path+"/"+str(x) + "_" + str(y) + ".png",image)
print('finish')









