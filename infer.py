import warnings

warnings.filterwarnings("ignore")


"""inference script for modular gtext"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K
import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import cv2

import sys


in_image = sys.argv[1]


ver = tf.__version__

def resizer(x,size,dataformat = "channels_last"):
    res = tf.image.resize_images(x,size,align_corners=True)
    return res

def resizer_block(tensor,size):
    layer = Lambda(lambda x : resizer(x,size))(tensor)
    return layer


loaded = tf.keras.models.load_model("model/keras.h5")
print(f"Model Loaded Successfully with tensoflow {ver}")

def predict_func(tensor):
    """custom flags"""
    
    BG = [0,0,0]
    FG = [255,255,255]
    
    sample_tensor = tensor
    load_it = load_img(sample_tensor,target_size=(512,512))
    array_it = img_to_array(load_it) / 255.
    
    array_it = array_it.astype(np.float32)
    expand_ = np.expand_dims(array_it,axis=0)
    prediction  = loaded.predict(expand_)
    
    
    maop = np.argmax(prediction>=0.98,axis=-1)
    maop = np.squeeze(maop)
    
    
    load_image = cv2.imread(sample_tensor)
    height ,width= load_image.shape[0],load_image.shape[1]
    
    newmask = scipy.ndimage.zoom(maop, (height/350,width/350), order=1, mode='nearest')
    maxes = newmask
    load_image[maxes==0] = BG
    
    load_image[maxes==1] = FG
    
    load_image[maxes==2] = FG
    
    load_image[maxes==3] = BG
    
    load_image[maxes==4] = BG
    
    return load_image
    
    
"""extended approach @"""
    
def TEXT_DETECT(file_,texts=14,kernal_size=(1,1)):
    H_param = 15
    file = file_
    mask_file = predict_func(file)
    image_  =cv2.cvtColor(mask_file,cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernal_size)
    erosion = cv2.dilate(image_,kernel,iterations = 1)
    image_ = cv2.Canny(erosion,127,255,0)
    canny_img = image_
    raw_image = cv2.imread(file,-1)
    cnts, hierarchy = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:texts] 
    rects = []
    rand_num = np.random.randint(0,1000,1)[0]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h >= H_param:
            rect = (x, y, w, h)
            rects.append(rect)
            rect_file = cv2.rectangle(raw_image, (x, y), (x+w, y+h), (0, 0, 255), 4)
            cv2.imwrite("output/"+str(rand_num)+"_gen.jpg",rect_file)
            
            
 
"""prediction function"""

PREDICT  = TEXT_DETECT(in_image)
