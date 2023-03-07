#!/usr/bin/env python
# coding: utf-8

# ## Illustrating table parsing by Squeeze-Unet and point set registration
# 
# ### Loading packages and the tableParser library


#from PIL import Image
import glob
import tqdm
import sys
import os
sys.path.append(os.path.abspath('Z:/faellesmappe/cmd/tfs/Sweden/SWE-DB/tableparsing/doccpd/doccpd'))
#sys.path.append(os.path.abspath('E:/faellesmappe/cmd/tfs/SWE-DB/tableparsing/doccpd/doccpd'))
#sys.path.append(os.path.abspath('C:/Users/sa-cmd/Documents/GitHub/SWE-DB/tableparsing/doccpd/doccpd'))
import pandas as pd
import cv2 as cv
import numpy as np

import tableParser

import pickle
import blosc
import itertools


from cv2_utils import erode,show


from skimage.filters import threshold_otsu

# def autocrop_noresize(image,width,height):
#     dim = (width,height)      
#     gray      = cv.cvtColor(image, cv.COLOR_BGR2GRAY)        
#     blur      = cv.GaussianBlur(gray.copy(),(5,5),0)
#     blur      = cv.medianBlur(gray.copy(),15,0)
#     tresh     = threshold_otsu(blur)    
#     _, threshold = cv.threshold(gray, tresh, 255, cv.THRESH_BINARY)
#     threshold = erode(threshold,(10,10),1)
    
#     contours, _  = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
#     # find the contour with the highest area, that will be
#     # a slightly too big crop of what we need
#     max_area = 0
#     for cnt in contours:
#         area = cv.contourArea(cnt)
#         if area > max_area:
#             max_area = area
#             best_cnt = cnt
            
#     # crop it like this so we can perform additional operations
#     # to further narrow down the crop
#     x, y, w, h = cv.boundingRect(best_cnt)
#     img=  image[y:y+h, x:x+w,:] 
  
#     return img

crop_info = {
      'top': 0.1,
      'bot': 0.5,
      'left': 0.01,
      'right':0.5
      }

crop_info = {
      'top': 0.1,
      'bot': 0.4,
      'left': 0.01,
      'right':0.85
      }


tableparsingmetricsdir  = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/tableparsing_summary/'
filebasedir             = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/minipics/'


crop_root               = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/minipics/Tabletype_A'
overlaydir              = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/overlay/Tabletype_A'
clouddir                = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/cloud/Tabletype_A'
template_image_path     = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/templates/A0032897_00549_cropped.jpg'

layout_pred = pd.read_csv('W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/output_predict_layout/preds.csv')
mask = layout_pred['pred']=='a'
typeA = layout_pred[mask]
filenames= typeA['filename_full'].to_list()
filenames = [os.path.basename(x) for x in filenames]
setfilenames = set([x.lower() for x in filenames])

pointclouddir = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/pointcloud/'
dat_files = glob.glob(pointclouddir+'/*.dat')

with open(dat_files[0], "rb") as f:
    compressed_pickle = f.read()
depressed_pickle = blosc.decompress(compressed_pickle)
data_full = pickle.loads(depressed_pickle)

for dat in dat_files[1:]: 
    with open(dat, "rb") as f:
        compressed_pickle = f.read()
    
    depressed_pickle = blosc.decompress(compressed_pickle)
    data1 = pickle.loads(depressed_pickle)

    data_full = list(itertools.chain(data_full, data1))


data_full_basename = [item + [os.path.basename(item[0])] for item in data_full]
data_full_new = list(filter(lambda x: x[2].lower() in setfilenames, data_full_basename))


template = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/templates/A0032897_00549_cropped_template.xml'
overlay = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/templates/A0032897_00549_cropped_overlay_deathdate.xml'   
evaluation = 'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/templates/A0032897_00549_cropped_evaluation_deathdate.xml'

begin = 0
end=1000
performance_metrics = []
for file,keypoints,_ in tqdm.tqdm(data_full_new[begin:end]):    
#for file,keypoints,_ in tqdm.tqdm(data_full_new[:2]): #(5,6,10)      
    print(file)
    try:
        IoU,dice,ownmetric = tableParser.tableParser_pointcloud(template_image_path=template_image_path,
                                        template=template,
                                        overlay=overlay,
                                        evaluation=evaluation,
                                        cropping=True,
                                        overlaydir=overlaydir,
                                        clouddir=clouddir,
                                        file=file,
                                        output=crop_root,                                        
                                        #output=None,
                                        keypoints=keypoints,
                                        crop_info=crop_info,
                                        detector=None,
                                        Rotating=None,
                                        jupyternotebook=False,
                                        show_fit=False,
                                        showcloud=False,
                                        showoverlay=False)
        print(file,ownmetric,(begin,end))
        performance_metrics.append([file,ownmetric])
   
   
    except:        
        performance_metrics.append([file,"Nan"])
 
with open(tableparsingmetricsdir+f'tableParsing_SWE_DB_Tabletype_A_deathdate_{str(begin)}-{str(end-1)}.pickle', 'wb') as handle:
    pickle.dump(performance_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
