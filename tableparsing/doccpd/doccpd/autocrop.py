# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:53:19 2022

@author: sa-cmd
"""
import cv2
from skimage.filters import threshold_otsu
from cv2_utils import erode,show

def autocrop(image,width,height):
    dim = (width,height)      
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    blur      = cv2.GaussianBlur(gray.copy(),(5,5),0)
    blur      = cv2.medianBlur(gray.copy(),15,0)
    tresh     = threshold_otsu(blur)    
    _, threshold = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)
    threshold = erode(threshold,(10,10),1)
    
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
    img=  image[y:y+h, x:x+w,:] 
                  
    img= cv2.resize(img.copy(),dim,interpolation = cv2.INTER_AREA)   
  
    return img

def autocrop_noresize(image,width,height):
    dim = (width,height)      
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    blur      = cv2.GaussianBlur(gray.copy(),(5,5),0)
    blur      = cv2.medianBlur(gray.copy(),15,0)
    tresh     = threshold_otsu(blur)    
    _, threshold = cv2.threshold(gray, tresh, 255, cv2.THRESH_BINARY)
#    threshold = erode(threshold,(10,10),1)
    
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
    img=  image[y:y+h, x:x+w,:] 
  
    return img

if __name__ == '__main__':
    files ='V:/BDADShareData2/SpecialClassProject/storage/examples/IMG_7693.jpg'
    width = int(2950)
    height = int(3871)
    try:
        original = cv2.imread(files,1)         
        crop     = autocrop(original,width,height)
        show(crop)        
    except:
        print(f'files did not work: {files}')
        pass
    cv.imwrite('V:/BDADShareData2/SpecialClassProject/storage/examples/cropIMG_7693.jpg',crop)