# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:16:18 2023

@author: sa-cmd
"""

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

#import tableParser

import pickle
import blosc
import itertools


from cv2_utils import erode,show

filebasedir             = 'Y:/RegionH/Scripts/data/storage/tableparsing_summary/TypeA'
pickle_files = glob.glob(filebasedir+'/*.pickle')

with open(pickle_files[0], "rb") as f:
    compressed_pickle = f.read()
data_full = pickle.loads(compressed_pickle)

for dat in pickle_files[1:]: 
    with open(dat, "rb") as f:
        compressed_pickle = f.read()    
    
    data1 = pickle.loads(compressed_pickle)

    data_full = list(itertools.chain(data_full, data1))
    
typeB_filenames = [x[0] for x in data_full if x[1]=="Not TypeA"]    
