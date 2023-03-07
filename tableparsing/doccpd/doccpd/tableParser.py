# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:07:20 2021

@author: sa-cmd
"""
import copy
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import tqdm
import glob
import gc
#import torch
import tensorflow as tf
tf.config.set_soft_device_placement(True)


from itertools import filterfalse

from PIL import Image
from IPython.display import display

import sys
sys.path.append(os.path.abspath('Z:/faellesmappe/cmd/tfs/cihvr-tsn/tableparsing/doccpd'))
#sys.path.append(os.path.abspath('Z:/faellesmappe/cmd/tfs/Sweden/SWE-DB/tableparsing/doccpd'))
#sys.path.append(os.path.abspath('Z:/faellesmappe/cmd/tfs/SpecialClassProject/tableparsing/doccpd/'))
#sys.path.append(os.path.abspath('E:/faellesmappe/cmd/tfs/SpecialClassProject/tableparsing/doccpd/'))
#cwd = os.getcwd()
#print('Current Working Directory is: ', cwd)
#os.chdir('Z:/faellesmappe/cmd/tfs/DeathCertificates1916_1920/doccpd/')

#cwd = os.getcwd()
#print('Current Working Directory is: ', cwd)
#os.chdir('../')

# from doccpd.template import Template, Overlay, crop_template
# from doccpd.keypoints import make_2D_detector_TF, make_1D_detector_TF,shift,KeypointDetector
# from doccpd.pointdrift import PointDrift
# from doccpd.pointcloud import PointCloud
# from doccpd.cv2_utils import show
# from doccpd.autocrop import autocrop
# from doccpd.cv2_utils import erode,show

from template import Template, Overlay, crop_template
from keypoints import make_2D_detector_TF, make_1D_detector_TF,shift,KeypointDetector
from pointdrift import PointDrift
from pointcloud import PointCloud
from cv2_utils import show
from autocrop import autocrop
from cv2_utils import erode,show

from itertools import filterfalse
from skimage.filters import threshold_otsu

def draw_points(
        image: np.ndarray,
        points: list or tuple,
        radius: int = 20,
        color: tuple = (255, 255, 255),
    ) -> np.ndarray:
    
    for p in points: # pylint: disable=C0103
        image = cv.circle(image, (int(p[0]), int(p[1])), radius, color, -1)

    return image

def MeanIOU_Dice(target_image,pdrift,keypoints,evaluation) -> np.ndarray:
    """
    ...
    """   
    
    mask1 = np.zeros(target_image.shape[:2])
    mask2 = np.zeros(target_image.shape)

    canvas_keypoints = keypoints.draw_on_image(mask1,size=10)
    drawn_evaluation = evaluation.point_cloud.draw_on_image(mask1)
    
    ymin = evaluation.point_cloud.points.min(axis=0)[1]
    xmin = evaluation.point_cloud.points.min(axis=0)[0]
    ymax = evaluation.point_cloud.points.max(axis=0)[1]
    xmax = evaluation.point_cloud.points.max(axis=0)[0]
    
    transformed      = pdrift.apply_transform(canvas_keypoints)
    transformed      = transformed[ymin:ymax,xmin:xmax].flatten()/255   

    drawn_evaluation   = drawn_evaluation[ymin:ymax,xmin:xmax].flatten()/255        
    
    intersection = np.sum(transformed*drawn_evaluation)
    total = np.sum(drawn_evaluation)+np.sum(transformed)
    union = total - intersection    
    
    IoU = intersection/union
    dice = 2*intersection/total
    ownmetric = intersection/np.sum(drawn_evaluation)
    
    return IoU,dice,ownmetric

def autocrop_noresize(image,width,height):
    dim = (width,height)      
    gray      = cv.cvtColor(image, cv.COLOR_BGR2GRAY)        
    blur      = cv.GaussianBlur(gray.copy(),(5,5),0)
    blur      = cv.medianBlur(gray.copy(),15,0)
    tresh     = threshold_otsu(blur)    
    _, threshold = cv.threshold(gray, tresh, 255, cv.THRESH_BINARY)
#    threshold = erode(threshold,(10,10),1)
    
    contours, _  = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # find the contour with the highest area, that will be
    # a slightly too big crop of what we need
    max_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
            
    # crop it like this so we can perform additional operations
    # to further narrow down the crop
    x, y, w, h = cv.boundingRect(best_cnt)
    img=  image[y:y+h, x:x+w,:] 
  
    return img


def SqueezeUnetModel(crop_info):
    detector = make_2D_detector_TF(
        vertical_model='W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/DeeplabV3PlusModel/SqueezeUnet_augmentation_All_color_Vertical.h5',
        vertical_ckpt= 'W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/tfcheckpoints_Vertical/SqueezeUnet_augmentation_All_color.ckpt',
        horizontal_model='W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/DeeplabV3PlusModel/SqueezeUnet_augmentation_All_color_Horizontal.h5',
        horizontal_ckpt=  'W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/tfcheckpoints_Horizontal/SqueezeUnet_augmentation_All_color.ckpt',
        crop_info=crop_info)
    return detector

def SqueezeUnetModel_horizontal(crop_info):
    detector = make_1D_detector_TF(
        model='W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/DeeplabV3PlusModel/SqueezeUnet_augmentation_All_color_Horizontal.h5',
        ckpt=  'W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/tfcheckpoints_Horizontal/SqueezeUnet_augmentation_All_color.ckpt',
        crop_info=crop_info)
    return detector

def SqueezeUnetModel_vertical(crop_info):
    detector = make_1D_detector_TF(
        model='W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/DeeplabV3PlusModel/SqueezeUnet_augmentation_All_color_Vertical.h5',
        ckpt= 'W:/BDADSharedData/NameNet/Sweden/storage/models/Unet/tfcheckpoints_Vertical/SqueezeUnet_augmentation_All_color.ckpt',
        crop_info=crop_info)
    return detector



def tableParser(template_image_path:str,
                template:str,
                overlay:str,
                overlaydir:str,
                clouddir:str,
                file:str,
                output:str,
                N_KEYPOINTS:int,
                crop_info:dict,
                detector,
                Rotating=None,
                cropping=False,
                show_fit=True,
                showoverlay=True,
                showcloud=True,
                jupyternotebook=False):
    
    template_image = cv.imdecode(np.fromfile(template_image_path, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    
    template = Template.from_xml(template, N_KEYPOINTS)
    cropped_temp = crop_template(template, crop_info)

    overlay = Overlay.from_xml(overlay)

    height, width = template_image.shape[:2]
    dim = (width,height)    
    
    try:        
        target_image = cv.imdecode(np.fromfile(file, dtype=np.uint8), cv.IMREAD_UNCHANGED)
        
        if Rotating==90:
            target_image = cv.rotate(target_image, cv.ROTATE_90_CLOCKWISE)
        if Rotating==270:    
            target_image = cv.rotate(target_image, cv.ROTATE_90_COUNTERCLOCKWISE)
        if Rotating==180:    
            target_image = cv.rotate(target_image, cv.ROTATE_180)
        
        if cropping:
            target_image = autocrop(target_image,height=height,width=width)
        else:    
            target_image = cv.resize(target_image, dim, interpolation = cv.INTER_AREA)
        
        keypoints    = detector.find_keypoints(target_image, N_KEYPOINTS,)
        pdrift       = PointDrift(cropped_temp)
        pdrift.fit(keypoints, iterations=50, show_fit=show_fit)   
        
        IoU,dice,ownmetric = MeanIOU_Dice(target_image,pdrift,keypoints,template)   
        
        transformed      = pdrift.apply_transform(target_image)
        drawn_overlay    = overlay.draw_on_image(transformed)
        canvas_keypoints = keypoints.draw_on_image(target_image,size=10)
        if output is not None and ownmetric>0.4:
            overlay.write_cells(transformed, file.split('\\')[-1].split('.')[0], root=output)
        if showoverlay:  
            if jupyternotebook:
                display(Image.fromarray(drawn_overlay).resize(size=(500,700)))
            else:
                show(drawn_overlay)
        if overlaydir is not None and ownmetric>0.4:    
            cv.imwrite(overlaydir+file.split('\\')[-1],drawn_overlay)
        if showcloud:  
            if jupyternotebook:
                display(Image.fromarray(canvas_keypoints))
            else:
                show(canvas_keypoints) 
        if clouddir is not None and ownmetric>0.4:  
            cv.imwrite(clouddir+file.split('\\')[-1],canvas_keypoints)   
        return IoU,dice,ownmetric,pdrift.q,pdrift.q/N_KEYPOINTS      
    except:
        print(f'{file} did not work')
        return "Nan","Nan","Nan","Nan","Nan"

def _all_pixels(mask):
    return cv.findNonZero(mask).squeeze()
def crop_pointcloud(target,keypoints, crop_info):
    pc = keypoints.points
    h, w = target.shape[:2]
    min_h = int(h * crop_info['top'])
    max_h = int(h * (1 - crop_info['bot']))
    min_w = int(w * crop_info['left'])
    max_w = int(w * (1 - crop_info['right']))    
    boundary = lambda x: (min_w < x[0] < max_w) and (min_h < x[1] < max_h)
    idxes = list(map(boundary, pc))
    pp = PointCloud(pc[idxes])
    return pp 

def tableParser_pointcloud(template_image_path:str,
                template:str,
                overlay:str,
                evaluation:str,
                overlaydir:str,
                clouddir:str,
                file:str,
                output:str,
                keypoints,
                crop_info:dict,
                threshold=0.05,
                detector=None,
                Rotating=None,
                cropping=False,
                show_fit=True,
                showoverlay=True,
                showcloud=True,
                jupyternotebook=False,
                autocrop=False):
    
    
    try:
        template_image = cv.imdecode(np.fromfile(template_image_path, dtype=np.uint8), cv.IMREAD_UNCHANGED)
        dim =(template_image.shape[1],template_image.shape[0]) 
        
        target_image = cv.imdecode(np.fromfile(file, dtype=np.uint8), cv.IMREAD_UNCHANGED)
        
        #target_image = cv.cvtColor(target_image, cv.COLOR_GRAY2BGR)
        h,w,_ = target_image.shape
        
        if h<3200 and h>2800:
        
            if autocrop:
                target_image=autocrop_noresize(target_image.copy(),height=h,width=w)           
                      
                mask = np.zeros_like(target_image)
                canvas_keypoints = keypoints.draw_on_image(mask,size=10)
                #canvas_keypoints = keypoints.draw_on_image(target_image,size=10)
                canvas_keypoints = cv.cvtColor(canvas_keypoints, cv.COLOR_BGR2GRAY)
               
                target_image= cv.resize(target_image.copy(),dim,interpolation = cv.INTER_AREA)  
                img2= cv.resize(canvas_keypoints,dim,interpolation = cv.INTER_AREA)
                kpimg2 = cv.findNonZero(img2).squeeze()        
                keypoints_res = PointCloud(points=kpimg2)
            else:
                keypoints_res=keypoints
            #template_keypoints = keypoints_res.draw_on_image(target_image,size=10)        
            
            keypoints_res = crop_pointcloud(target_image, keypoints_res,crop_info)
            if len(keypoints_res.points)>20000:
                idx = np.random.choice(len(keypoints_res.points), size=20000, replace=False)
                keypoints_res = PointCloud(points=keypoints_res.points[idx,:])
            
            template = Template.from_xml(template, 20000)
            evaluation = Template.from_xml(evaluation, 20000)
            #evaluation_keypoints = evaluation.point_cloud.draw_on_image(template_image,size=10)
            #show(evaluation_keypoints)
            cropped_temp = crop_template(template, crop_info)
            overlay = Overlay.from_xml(overlay)
        
                       
            pdrift = PointDrift(cropped_temp)
            #pdrift.fit(keypoints_res, iterations=50, show_fit=False)
            pdrift.fit(keypoints_res, iterations=50, show_fit=show_fit)
                                    
            IoU,dice,ownmetric = MeanIOU_Dice(target_image,pdrift,keypoints_res,evaluation)   
            
            transformed      = pdrift.apply_transform(target_image)
            drawn_overlay    = overlay.draw_on_image(transformed)
            canvas_keypoints = keypoints_res.draw_on_image(target_image,size=10)
            #show(canvas_keypoints)       
            #show(drawn_overlay)
            #drawn_evaluation    = evaluation.draw_on_image(transformed)
            #show(canvas_keypoints)       
            #print(os.path.join(overlaydir,os.path.basename(file))
            #print(os.path.exists(os.path.join(overlaydir,os.path.basename(file))))
            if output is not None:
                subdirs = os.listdir(output)
                if len(subdirs)>0:
                    alreadycopy = os.path.exists(os.path.join(output,subdirs[0],os.path.basename(file)))
                else:    
                    alreadycopy=False
            if output is not None and ownmetric>threshold:
                if alreadycopy:
                    overlay.write_cells(transformed, file.split('/')[-1].split('.')[0], root=output)
                else:                
                    overlay.write_cells(transformed, file.split('/')[-1].split('.')[0], root=output)
            if showoverlay:  
                if jupyternotebook:
                    display(Image.fromarray(drawn_overlay).resize(size=(500,700)))
                else:
                    show(drawn_overlay)
            if overlaydir is not None and ownmetric>threshold:    
                if alreadycopy:
                    cv.imwrite(overlaydir+'/'+file.split('/')[-1],drawn_overlay)
                else:    
                    cv.imwrite(overlaydir+'/'+file.split('/')[-1],drawn_overlay)
            if showcloud:  
                if jupyternotebook:
                    display(Image.fromarray(canvas_keypoints))
                else:
                    show(canvas_keypoints) 
            if clouddir is not None and ownmetric>threshold:   
                if alreadycopy:
                    cv.imwrite(clouddir+'/'+file.split('/')[-1],canvas_keypoints) 
                else:    
                    cv.imwrite(clouddir+'/'+file.split('/')[-1],canvas_keypoints)   
            return IoU,dice,ownmetric  
        else:
            return "Not TypeA","Not TypeA","Not TypeA"
    except:
        print(f'{file} did not work')
        return "Nan","Nan","Nan"
        

def tableParser_pointcloud_local(template_image_path:str,
                template:str,
                overlay_list:list,
                evaluation_list:list,
                overlaydir:str,
                clouddir:str,
                file:str,
                output:str,
                keypoints,
                crop_info_list:list,
                threshold=0.1,
                detector=None,
                Rotating=None,
                cropping=False,
                show_fit=True,
                showoverlay=True,
                showcloud=True,
                jupyternotebook=False):
    
    
    try:
        template_image = cv.imdecode(np.fromfile(template_image_path, dtype=np.uint8), cv.IMREAD_UNCHANGED)
        dim =(template_image.shape[1],template_image.shape[0]) 
        
        target_image = cv.imdecode(np.fromfile(file, dtype=np.uint8), cv.IMREAD_UNCHANGED)  
        
#        target_image = cv.cvtColor(target_image, cv.COLOR_GRAY2BGR)
        h,w,_ = target_image.shape
        
        target_image=autocrop_noresize(target_image.copy(),height=h,width=w)
#        target_image = target_image[:,40:,:] # crop target image

        mask = np.zeros_like(target_image)
        canvas_keypoints = keypoints.draw_on_image(mask,size=10)
        #canvas_keypoints = keypoints.draw_on_image(target_image,size=10)
        canvas_keypoints = cv.cvtColor(canvas_keypoints, cv.COLOR_BGR2GRAY)
       
        target_image= cv.resize(target_image.copy(),dim,interpolation = cv.INTER_AREA)  
        
        img2= cv.resize(canvas_keypoints,dim,interpolation = cv.INTER_AREA)
        kpimg2 = cv.findNonZero(img2).squeeze()        
        keypoints_res_all = PointCloud(points=kpimg2)
        #template_keypoints = keypoints_res.draw_on_image(target_image,size=10)        
        template_all = Template.from_xml(template, 20000)        
        IoU,dice,ownmetric = [],[],[]
        for crop_info,overlay_,evaluation_ in zip(crop_info_list,overlay_list,evaluation_list):
        #for crop_info in crop_info_list:
            template_all_copy= copy.copy(template_all)
            print(crop_info)
            #print(overlay_)
            #print(evaluation_)
            keypoints_res = crop_pointcloud(target_image, keypoints_res_all,crop_info)
            if len(keypoints_res.points)>20000:
                idx = np.random.choice(len(keypoints_res.points), size=20000, replace=False)
                keypoints_res = PointCloud(points=keypoints_res.points[idx,:])
            cropped_temp = crop_template(template_all_copy, crop_info)
            
                        
            pdrift=0
            pdrift = PointDrift(cropped_temp)        
            pdrift.fit(keypoints_res, iterations=50, show_fit=show_fit)        
                        
            evaluation = Template.from_xml(evaluation_, 20000)
            IoU_,dice_,ownmetric_ = MeanIOU_Dice(target_image,pdrift,keypoints_res,evaluation)   
                
            transformed      = pdrift.apply_transform(target_image)
            
            overlay = Overlay.from_xml(overlay_)        
            drawn_overlay    = overlay.draw_on_image(transformed)
#            drawn_overlay    = overlay.draw_on_image(template_image)
                        
            canvas_keypoints = keypoints_res.draw_on_image(target_image,size=10)
#            show(canvas_keypoints)
           # show(drawn_overlay)
            if output is not None:
                subdirs = os.listdir(output)
                if len(subdirs)>0:
                    alreadycopy = os.path.exists(os.path.join(output,subdirs[0],os.path.basename(file)))
                else:    
                    alreadycopy=False
                alreadycopy=False    
            if output is not None and ownmetric_>threshold:
                if alreadycopy:
                    overlay.write_cells(transformed, file.split('\\')[-1].split('.')[0], root=output)
                else: 
                    overlay.write_cells(transformed, file.split('\\')[-1].split('.')[0], root=output)
            if showoverlay:  
                if jupyternotebook:
                    display(Image.fromarray(drawn_overlay).resize(size=(500,700)))
                else:
                    show(drawn_overlay)
            if overlaydir is not None and ownmetric_>threshold:    
                if alreadycopy:
                    cv.imwrite(overlaydir+'/'+file.split('\\')[-2]+'_'+file.split('\\')[-1],drawn_overlay)
                else:    
                    cv.imwrite(overlaydir+'/'+file.split('\\')[-1],drawn_overlay)
            if showcloud:  
                if jupyternotebook:
                    display(Image.fromarray(canvas_keypoints))
                else:
                    show(canvas_keypoints) 
            if clouddir is not None and ownmetric_>threshold:   
                if alreadycopy:
                    cv.imwrite(clouddir+'/'+file.split('\\')[-2]+'_'+file.split('\\')[-1],canvas_keypoints) 
                else:    
                    cv.imwrite(clouddir+'/'+file.split('\\')[-1],canvas_keypoints)   
            IoU.append(IoU_)
            dice.append(dice_)
            ownmetric.append(ownmetric_)
        return IoU,dice,ownmetric              
    except:
        print(f'{file} did not work')
        return "Nan","Nan","Nan"


def UNET_FIT(   file:str,
                output:str,
                N_KEYPOINTS:int,
                crop_info:dict,                
                clouddir:str,
                detector,
                Rotating=None,
                cropping=False,
                showcloud=True,
                jupyternotebook=False):
    
    try:        
        target_image = cv.imdecode(np.fromfile(file, dtype=np.uint8), cv.IMREAD_UNCHANGED)
        height, width = target_image.shape[:2]
        
        if Rotating==90:
            target_image = cv.rotate(target_image, cv.ROTATE_90_CLOCKWISE)
        if Rotating==270:    
            target_image = cv.rotate(target_image, cv.ROTATE_90_COUNTERCLOCKWISE)
        if Rotating==180:    
            target_image = cv.rotate(target_image, cv.ROTATE_180)
        
        if cropping:
            target_image = autocrop(target_image,height=height,width=width)
        
        keypoints    = detector.find_keypoints(target_image, N_KEYPOINTS,)

        if showcloud:  
            canvas_keypoints = keypoints.draw_on_image(target_image,size=10)
            if jupyternotebook:
                display(Image.fromarray(canvas_keypoints))
            else:
                show(canvas_keypoints) 
        if clouddir is not None:  
            cv.imwrite(clouddir+file.split('\\')[-1],canvas_keypoints)   
        return keypoints
    except:
        print(f'{file} did not work')
        return "Nan"
    


if __name__ == '__main__':
      

    N_KEYPOINTS = 10_000
    crop_info = {
      'top': 0.01,
      'bot': 0.6,
      'left': 0.01,
      'right':0.05
      }
    
    
    crop_root= 'V:/BDADShareData2/SpecialClassProject/storage/minipics'
    overlaydir = 'V:/BDADShareData2/SpecialClassProject/storage/overlay/'
    clouddir = 'V:/BDADShareData2/SpecialClassProject/storage/cloud/'
    template_image_path='V:/BDADShareData2/SpecialClassProject/templates/cropIMG_7693.jpg'
    
    template = 'V:/BDADShareData2/SpecialClassProject/templates/cropIMG_7693_template.xml'
    template = 'V:/BDADShareData2/SpecialClassProject/templates/D-Engström_scan_2020-04-30_10-28-00_page_1_template.xml'
    overlay = 'V:/BDADShareData2/SpecialClassProject/templates/D-Engström_scan_2020-04-30_10-28-00_page_1_overlay.xml'
    detector       = SqueezeUnetModel(crop_info)
    image = 'V:/BDADShareData2/SpecialClassProject/1941/04.25-05.25/F8592-8602\\IMG_8592.JPG'    #(ex2)
    image = 'V:/BDADShareData2/SpecialClassProject/1941/04.25-05.25/F8504-8521/IMG_8504.JPG'    
    image = 'V:/BDADShareData2/SpecialClassProject/1941/04.25-05.25/F8603-8617\\IMG_8603.JPG'
    
    image = 'V:/BDADShareData2/SpecialClassProject/1941/04.25-05.25/F8625-8633\\IMG_8625.JPG'
    
    image = 'V:/BDADShareData2/SpecialClassProject/jpg/1928\\AB-Andersson_scan_2020-05-06_13-53-12_page_1.jpg'
    image = 'V:/BDADShareData2/SpecialClassProject/jpg/1928\\2020-05-07_177-185 (IQ)_Bild_177-konverterad_page_1.jpg'
    
    image= 'V:/BDADShareData2/SpecialClassProject/jpg/1928\\D-Engström_scan_2020-04-30_10-28-00_page_1.jpg'   
    image = 'V:/BDADShareData2/SpecialClassProject/jpg/1928/D-Engström_scan_2020-05-04_12-40-35_page_1.jpg'
    image = 'V:/BDADShareData2/SpecialClassProject/jpg/1928/D-Engström_scan_2020-05-04_12-33-44_page_1.jpg'
    #image ='V:/BDADShareData2/SpecialClassProject/jpg/1928/2020-05-07_177-185 (IQ)_Bild_177-konverterad_page_1.jpg'
    image = 'V:/BDADShareData2/SpecialClassProject/jpg/1928/Kallander-Karlsson St_scan_2020-05-18_13-26-44_page_1.jpg'

    image = 'V:/BDADShareData2/SpecialClassProject/jpg/1928/Kallander-Karlsson St_scan_2020-05-18_13-28-09_page_1.jpg'
    image = 'V:/BDADShareData2/SpecialClassProject/jpg/1928/Kallander-Karlsson St_scan_2020-05-18_13-33-03_page_1.jpg'  # (ex1)
    for i in [None]:
        IoU,dice,ownmetric,logL,logLmean = tableParser(
                                        template_image_path=template_image_path,
                                        template=template,
                                        overlay=overlay,
                                        cropping=True,
                                        overlaydir=None,
                                        clouddir=None,
                                        file=image,
                                        output=None,
                                        N_KEYPOINTS=10000,
                                        crop_info=crop_info,
                                        detector=detector,
                                        Rotating=i,
                                        jupyternotebook=False,
                                        show_fit=True,
                                        showcloud=True)
        print(i,[IoU,dice,ownmetric,logL,logLmean])
        
    #file=image        
    #img = cv.imdecode(np.fromfile(image, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    #show(img)       
# W:/BDADSharedData/FilesForInstallation/Capture1.JPG 

# V:/BDADShareData2/SpecialClassProject/jpg/1928/Sa-Si_scan_2020-05-28_14-14-23_page_6.jpg
# W:/BDADSharedData/FilesForInstallation/Capture.JPG
#dd = 'V:/BDADShareData2/SpecialClassProject/jpg/1928/Ljung-Lö_scan_2020-05-20_11-05-55_page_1.jpg'
#target_image = cv.imread(dd, 1)
#show(target_image)
