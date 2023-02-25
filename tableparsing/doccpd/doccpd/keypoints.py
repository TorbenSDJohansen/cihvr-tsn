from abc import abstractmethod
from collections.abc import Iterable
import itertools

import cv2 as cv
import numpy as np
#import torch

from PIL import Image
from skimage.morphology import skeletonize

import sys
import os
#sys.path.append(os.path.abspath('E:/faellesmappe/cmd/tfs/SpecialClassProject/tableparsing/doccpd/'))
from utils import flatten_list
#from doccpd.unet_model import UNet
from pointcloud import PointCloud

from cv2_utils import show

import tensorflow as tf
from tensorflow.keras.models import load_model



def _nonzero_coordinates(array): return cv.findNonZero(array)


def _mask_to_image(mask): return (mask * 255).astype(np.uint8)


def _clsname(cls): return type(cls).__name__


class KeypointBackend:

    @abstractmethod
    def identify_mask(self): raise NotImplementedError()

    def __repr__(self):
        return f'{_clsname(self)}'

class TableNet1D_TF(KeypointBackend):
    
    def __init__(self, model=None, ckpt=None):
        self._model = load_model(model)
        self._model.load_weights(ckpt)        
        self._IMAGE_SIZE=448

    def __repr__(self):
        return f'{_clsname(self)} | device = {self._device}, threshold = {self._prediction_threshold}'

    def identify_mask(self, image):        
        processed = self._preprocess_image(image)
        ny_image,nx_image,nz_image = image.shape
        
        _,target_height,target_width,__ = processed.shape
        target_height = target_height/self._IMAGE_SIZE
        target_width  = target_width/self._IMAGE_SIZE
        
        patches = tf.image.extract_patches(images=processed,
                           sizes=[1, self._IMAGE_SIZE, self._IMAGE_SIZE, 1],
                           strides=[1,self._IMAGE_SIZE, self._IMAGE_SIZE,1],
                           rates=[1, 1, 1, 1],
                           padding='VALID')
        patches_re = tf.reshape(patches, [int(target_height)*int(target_width), self._IMAGE_SIZE, self._IMAGE_SIZE, 3])

        image_predict  = []

        for jter in range(0,patches_re.shape[0]):
            mask_predict = self.infer(image=patches_re[jter])             
            #image_predict.append(tf.where(mask_predict==1,int(255),int(0)).numpy().astype(np.uint8))    
            #image_predict.append(tf.where(mask_predict==2,int(255),int(0)).numpy().astype(np.uint8))    
            image1 = tf.where(mask_predict==1,int(255),int(0))                         
            #image2 = tf.where(mask_predict==2,int(175),int(0))            
            #image2 = tf.where(mask_predict==2,int(0),int(0))
            #imageT = tf.math.add_n([image1,image2]).numpy().astype(np.uint8)            
            imageT=image1.numpy().astype(np.uint8)
            image_predict.append(imageT)    
            
    
        rows           = tf.split(tf.expand_dims(tf.convert_to_tensor(image_predict),axis=3),int(target_height),axis=0)
        rows           = [tf.concat(tf.unstack(x),axis=1) for x in rows] 
        reconstructed  = tf.concat(rows,axis=0)
 #      reconstructed  = tf.image.resize_with_crop_or_pad(reconstructed,ny_image,nx_image).numpy().astype(np.uint8)
        reconstructed  = tf.image.resize_with_crop_or_pad(reconstructed,ny_image,nx_image)
        padding        = int(255) - tf.image.resize_with_crop_or_pad(tf.ones_like(reconstructed),int(ny_image),int(nx_image)) * tf.constant(int(255))
        reconstructed += padding
        reconstructed = reconstructed.numpy().astype(np.uint8)
        reconstructed[-10:,:]=int(0)
        reconstructed[:10:,:]=int(0)
        return  reconstructed

    def infer(self,image):
        predictions = self._model.predict(np.expand_dims((image), axis=0),verbose=0)
        #predictions = self._model.predict(image)
        predictions = np.squeeze(predictions)
        predictions = np.argmax(predictions, axis=2)
        return predictions

        
    @staticmethod
    def _preprocess_image(image):
        _IMAGE_SIZE=448
        #image        = tf.convert_to_tensor(image, dtype=tf.float32) # If image is loaded by cv2.imread(imagefile)
        #image        = tf.image.decode_jpeg(image, channels=3) # If image is loaded by tf.io.read_file(imagefile)
        #image        = image/127 - 1
        image        = np.expand_dims((image), axis=0)
        target_height, target_width = tf.math.ceil(image.shape[1]/_IMAGE_SIZE),tf.math.ceil(image.shape[2]/_IMAGE_SIZE)
        image                       = tf.image.resize_with_crop_or_pad(image, 
                                                               int(target_height*_IMAGE_SIZE), 
                                                               int(target_width*_IMAGE_SIZE))
        
        padding = 255 - tf.image.resize_with_crop_or_pad(tf.ones_like(image),int(target_height*_IMAGE_SIZE),int(target_width*_IMAGE_SIZE)) * tf.constant(255, dtype=tf.uint8)
        
        image = image + padding
        return (image/127-1)
    

class TableNet2D_TF(KeypointBackend):
    def __init__(self, vertical_model=None, vertical_ckpt=None, horizontal_model=None, horizontal_ckpt=None):
        self._v_model = TableNet1D_TF(vertical_model,vertical_ckpt)
        self._h_model = TableNet1D_TF(horizontal_model,horizontal_ckpt)
        
    def identify_mask(self, image):
        v_mask = self._v_model.identify_mask(image)
        h_mask = self._h_model.identify_mask(image)
        return cv.add(v_mask, h_mask)    


class KeypointDetector:

    def __init__(self, backend=None, crop_info=None):
        self._backend = backend or TableNet1D_TF()
        self._crop_info = crop_info
        self._device = self._backend._device

    def __repr__(self):
        return f'{_clsname(self)} | Backend = ({self._backend})'

    def find_lines(self, image):
        mask = self._backend.identify_mask(image)
        skeleton = _isolate_lines(mask)
        return _find_contours(skeleton)

    def find_all_pixels(self, image):
        mask = self._backend.identify_mask(image)
        skeleton = _extract_skeleton(mask)
        return _all_pixels(skeleton)

    def find_keypoints(self, image, sample_size):
        image = np.array(image)
        if self._crop_info:
            image = crop(image, self._crop_info)
        keypoints = self.find_all_pixels(image)
        distributed = _get_evenly_spaced_numbers(
                stop=len(keypoints)-1,
                n=min(sample_size, len(keypoints)-1)
        )
        numbers = PointCloud(keypoints[distributed])
        shifted = shift(numbers, self._crop_info, image.shape[1], image.shape[0])
        return shifted

class KeypointDetector_TF:

    def __init__(self, backend=None, crop_info=None):
        self._backend = backend or TableNet1D_TF()
        self._crop_info = crop_info
        
    def __repr__(self):
        return f'{_clsname(self)} | Backend = ({self._backend})'

    def find_lines(self, image):
        mask = self._backend.identify_mask(image)
        skeleton = _isolate_lines(mask)
        return _find_contours(skeleton)

    def find_all_pixels(self, image):
        mask = self._backend.identify_mask(image)
        skeleton = _extract_skeleton(mask)
        return _all_pixels(skeleton)

    def find_keypoints(self, image, sample_size):
        image = np.array(image)
        if self._crop_info:
            image = crop(image, self._crop_info)
        keypoints = self.find_all_pixels(image)
        distributed = _get_evenly_spaced_numbers(
                stop=len(keypoints)-1,
                n=min(sample_size, len(keypoints)-1)
        )
        numbers = PointCloud(keypoints[distributed])
        shifted = shift(numbers, self._crop_info, image.shape[1], image.shape[0])
        return shifted


def crop(image, crop_info):
    height, width = image.shape[:2]

    min_height = int(height * crop_info['top'])
    max_height = int(height * (1 - crop_info['bot']))
    min_width = int(width * crop_info['left'])
    max_width = int(width * (1 - crop_info['right']))

    return image[min_height:max_height, min_width:max_width]


def shift(point_cloud, crop_info, width, height):
    points = point_cloud.points # TODO: Make PointCloud a class. Too much responsibility
    shifted = np.zeros_like(points)
    shifted[:, 1] = points[:, 1] + crop_info['top'] / (1 - crop_info['top'] - crop_info['bot']) * height
    shifted[:, 0] = points[:, 0] + crop_info['left'] / (1 - crop_info['left'] - crop_info['right']) * width
    return PointCloud(shifted)
   
def make_1D_detector_TF(model=None, ckpt=None, crop_info=None):
    return KeypointDetector_TF(TableNet1D_TF(model=model, ckpt=ckpt), crop_info)

def make_2D_detector_TF(vertical_model=None, vertical_ckpt=None, horizontal_model=None, horizontal_ckpt=None, crop_info=None):
    return KeypointDetector_TF(TableNet2D_TF(vertical_model, vertical_ckpt, horizontal_model, horizontal_ckpt), crop_info)

def _aspect_ratio(contour): 
    _, _, w, h = cv.boundingRect(contour)
    return float(w) / h


def _is_line(contour): return _aspect_ratio(contour) < 0.2


def _find_contours(image): 
    return cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]


def _skeletonize(mask): return skeletonize(cv.divide(mask, 255))


def _extract_skeleton(mask): return np.array(_skeletonize(mask) * 255., dtype=np.uint8)


def _sort_by_idx_1(array): return array[array[:, 1].argsort()]


def _get_evenly_spaced_numbers(stop, n):
    return np.floor(np.linspace(0, stop, n)).astype(int)


def _isolate_lines(mask):
    canvas = np.zeros_like(mask)
    cnts = _find_contours(mask)
    gc = [c for c in cnts if _is_line(c)]
    cv.drawContours(canvas, gc, -1, 255, -1)
    return _extract_skeleton(canvas)


def _all_pixels(mask):
    return cv.findNonZero(mask).squeeze()
