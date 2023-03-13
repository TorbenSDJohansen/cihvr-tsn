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
#sys.path.append(os.path.abspath('Z:/faellesmappe/cmd/tfs/cihvr-tsn/tableparsing/doccpd/doccpd'))
#sys.path.append(os.path.abspath('E:/faellesmappe/cmd/tfs/SWE-DB/tableparsing/doccpd/doccpd'))
sys.path.append(os.path.abspath('C:/Users/sa-cmd/Documents/GitHub/cihvr-tsn/tableparsing/doccpd/doccpd'))
import pandas as pd
import cv2 as cv
import numpy as np

import tableParser

import pickle
import blosc
import itertools


from cv2_utils import erode,show


from skimage.filters import threshold_otsu

crop_info = {
      'top': 0.55,
      'bot': 0.2,
      'left': 0.01,
      'right':0.01
      }

filebasedir             = 'Y:/RegionH/Scripts/data/storage/'

tableparsingmetricsdir  = filebasedir + 'tableparsing_summary/TypeA_bottom_smaller_crop/'
crop_root = 'V:/BDADShareData2/cihvr/data/storage/minipics/TypeA_bottom_smaller_crop'
#crop_root               = filebasedir+'minipics/TypeA_bottom_smaller_crop'
overlaydir              = filebasedir+'overlay/TypeA_bottom_smaller_crop'
clouddir                = filebasedir+'cloud/TypeA_bottom_smaller_crop'
template_image_path     = 'Y:/RegionH/Scripts/data/templates_and_overlays/TypeA/SP2_00004.pdf.page-0.jpg'

pointclouddir = 'Y:/RegionH/Scripts/data/pointclouds_tableb/'
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


template = 'Y:/RegionH/Scripts/data/templates_and_overlays/TypeA/SP2_00004.pdf.page-0_template.xml'
overlay = 'Y:/RegionH/Scripts/data/templates_and_overlays/TypeA/SP2_00004.pdf.page-0_overlay_bottom.xml'   
evaluation = template #'W:/BDADSharedData/Spanish Flu/Sweden/storage_additionalAdrian/templates/A0032897_00549_cropped_evaluation_deathdate.xml'

begin = 90000+3845
end=len(data_full)
performance_metrics = []
for file,keypoints in tqdm.tqdm(data_full[begin:end]):    
#for file,keypoints in tqdm.tqdm(data_full[:10]): #(5,6,10)      
    print(file)
    try:
        IoU,dice,ownmetric = tableParser.tableParser_pointcloud(template_image_path=template_image_path,
                                        template=template,
                                        overlay=overlay,
                                        evaluation=evaluation,
                                        cropping=False,
                                        overlaydir=None,
                                        clouddir=None,
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
 
with open(tableparsingmetricsdir+f'tableParsing_CIHVR_TypeA_bottom_{str(begin)}-{str(end-1)}.pickle', 'wb') as handle:
    pickle.dump(performance_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
