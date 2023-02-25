# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:07:20 2021

@author: sa-cmd
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2 as cv
import matplotlib.pyplot as plt

import pickle
import blosc

from PIL import Image
import gc
import tensorflow as tf
tf.config.set_soft_device_placement(True)

import glob
import pickle
import tqdm
import sys
import os



sys.path.append(os.path.abspath('Z:/faellesmappe/cmd/tfs/cihvr-tsn/tableparsing/doccpd'))
sys.path.append(os.path.abspath('Z:/faellesmappe/cmd/tfs/cihvr-tsn/data'))
#sys.path.append(os.path.abspath('E:/faellesmappe/cmd/tfs/DK-Census/tableparsing/doccpd'))
#sys.path.append(os.path.abspath('C:/Users/sa-sfw/Documents/GitHub/DK-Census/tableparsing/doccpd/'))


from table_b_pages import get_table_b_pages

map_images_ds = get_table_b_pages()




import pandas as pd
import cv2 as cv
import numpy as np

from doccpd.tableParser import SqueezeUnetModel
from doccpd.template import Template, Overlay, crop_template
from doccpd.keypoints import make_2D_detector_TF, shift
from doccpd.pointdrift import PointDrift
from doccpd.pointcloud import PointCloud
from doccpd.cv2_utils import show
from doccpd.autocrop import autocrop_noresize

pointclouddir = 'Y:/RegionH/Scripts/data/pointclouds_tableb/'
    
crop_info = {
      'top': 0.0,
      'bot': 0.0,
      'left': 0.0,
      'right':0.0
      }
detector       = SqueezeUnetModel(crop_info)    


#filenames = glob.glob('W:/BDADSharedData/Spanish Flu/Sweden/AdditionalAdrian/*.jpg')
filenames = []
for fname, basename in map_images_ds.items():
    filenames.append(fname)
N_KEYPOINTS = 20_000

detector = make_2D_detector_TF(
    vertical_model='W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/DeeplabV3PlusModel/SqueezeUnet_augmentation_All_color_Vertical.h5',
    vertical_ckpt= 'W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/tfcheckpoints_Vertical/SqueezeUnet_augmentation_All_color.ckpt',
    horizontal_model='W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/DeeplabV3PlusModel/SqueezeUnet_augmentation_All_color_Horizontal.h5',
    horizontal_ckpt=  'W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/tfcheckpoints_Horizontal/SqueezeUnet_augmentation_All_color.ckpt',
    crop_info=crop_info)

# Below works for the residual...sorry poor programming/patching
'''
print("Loading from Checkpoint")
keypointlist = []

## This part because the server stop during the itereation
with open(pointclouddir+"pointcloud_with_cropping.dat", "rb") as f:
    compressed_pickle = f.read()
depressed_pickle = blosc.decompress(compressed_pickle)
keypointlist = pickle.loads(depressed_pickle)

    
startindex_afterserverbreakdown = filenames.index(keypointlist[-1][0])+1

'''
keypointlist = []

start =0
end = 50

n = 0
for key in tqdm.tqdm(filenames[start:end]):  
        try:
            n +=1
            target_image = cv.imdecode(np.fromfile(key, dtype=np.uint8), cv.IMREAD_UNCHANGED) 
            #target_image = cv.cvtColor(target_image, cv.COLOR_GRAY2BGR)
            h,w,_ = target_image.shape
            target_imagec = autocrop_noresize(target_image.copy(),height=h,width=w)            
            keypoints    = detector.find_keypoints(target_imagec, N_KEYPOINTS,) 
            
#            canvas_keypoints = keypoints.draw_on_image(target_imagec,size=10)
 #           cv.imwrite('W:/BDADSharedData/Spanish Flu/Denmark/census1916/storagecph/cloud/'+key,canvas_keypoints) 
            
            keypointlist.append([key,keypoints])                
            if n % 1000 == 0:  
                print('saving results')
                pickled_data = pickle.dumps(keypointlist)  # returns data as a bytes object
                compressed_pickle = blosc.compress(pickled_data)
                with open(pointclouddir+"pointcloud_with_cropping_"+str(n+start)+".dat", "wb") as f:
                    f.write(compressed_pickle)
                
                pickled_data, compressed_pickle, keypointlist = 0, 0, []

        except:
            print(f'{key} did not work')

pickled_data = pickle.dumps(keypointlist)  # returns data as a bytes object
compressed_pickle = blosc.compress(pickled_data)

with open(pointclouddir+"pointcloud_with_cropping"+str(n+start)+".dat", "wb") as f:
    f.write(compressed_pickle)

'''
How to open the compressed file
with open(pointclouddir+"pointcloud_with_cropping.dat", "rb") as f:
    compressed_pickle = f.read()

depressed_pickle = blosc.decompress(compressed_pickle)
data = pickle.loads(depressed_pickle)

'''
