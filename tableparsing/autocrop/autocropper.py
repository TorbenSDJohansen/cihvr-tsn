# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:35:10 2020

@author: sa-cmd
"""

import cv2
from skimage.filters import threshold_otsu


def autocrop(image):      
    # apply a tolerance on a gray version of the image to
    # select the non-black pixels
   # image=original.copy()
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    blur      = cv2.GaussianBlur(gray.copy(),(5,5),0)
    tresh     = threshold_otsu(blur)
    _, threshold = cv2.threshold(gray, tresh*1.2, 255, cv2.THRESH_BINARY)
    contours, _  = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # find the contour with the highest area, that will be
    # a slightly too big crop of what we need
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
            
    # crop it like this so we can perform additional operations
    # to further narrow down the crop
    x, y, w, h = cv2.boundingRect(best_cnt)
    gray_crop  = gray[y:y+h, x:x+w]                
    im_h, im_w = gray.shape[:2]
    scale_h, scale_w = 1.0, 1.0
    img= cv2.resize(gray_crop, (int(im_w * scale_w), int(im_h * scale_h)), interpolation = cv2.INTER_AREA)        
  
    return img
