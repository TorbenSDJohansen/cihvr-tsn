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
import numbers


from cv2_utils import erode,show

filebasedir             = 'Y:/RegionH/Scripts/data/storage/tableparsing_summary/TypeB/'
pickle_files_TypeB2 = glob.glob(filebasedir+'/tableParsing_CIHVR_TypeB2_bottom_*.pickle')
pickle_files_TypeB = glob.glob(filebasedir+'/tableParsing_CIHVR_TypeB_bottom_*.pickle')

with open(pickle_files_TypeB2[0], "rb") as f:
    compressed_pickle = f.read()
data_full_TypeB2 = pickle.loads(compressed_pickle)

for dat in pickle_files_TypeB2[1:]: 
    with open(dat, "rb") as f:
        compressed_pickle = f.read()    
    
    data1 = pickle.loads(compressed_pickle)

    data_full_TypeB2 = list(itertools.chain(data_full_TypeB2, data1))
    
data_full_TypeB2  = [x for x in data_full_TypeB2 if isinstance(x[1],numbers.Number)]    
TypeB2_filenames  = [x[0] for x in data_full_TypeB2 if x[1]>0.75]    

with open(pickle_files_TypeB[0], "rb") as f:
    compressed_pickle = f.read()
data_full_TypeB = pickle.loads(compressed_pickle)

for dat in pickle_files_TypeB[1:]: 
    with open(dat, "rb") as f:
        compressed_pickle = f.read()    
    
    data1 = pickle.loads(compressed_pickle)

    data_full_TypeB = list(itertools.chain(data_full_TypeB, data1))

    
data_full_TypeB  = [x for x in data_full_TypeB if isinstance(x[1],numbers.Number)]    
TypeB_filenames  = [x[0] for x in data_full_TypeB if x[1]>0.82]  

## Finding longtables for Torben
#TypeB2_filenames  = set([x[0] for x in data_full_TypeB2 if x[1]<0.3])
#TypeB_filenames  = set([x[0] for x in data_full_TypeB if x[1]<0.3])    

#longtables = TypeB2_filenames.intersection(TypeB_filenames)
