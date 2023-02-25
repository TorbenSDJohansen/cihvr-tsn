# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:33:32 2020

@author: sa-cmd
"""


import sys
import os

import cv2

from autocropper import autocrop



croppers = [
#8010482341,
#8010750191,
#8010750661,
#8010755501,
8010755551,
"8010755581-6486799",
"8010755581-6486800",
"8010755581-21047215",
"8010755641-6486802",
"8010755641-21047216",
8010755661,
8010755831,
8010755861,
"8010755871-6486831",
"8010755871-21047220",
8010755891,
8010759891,
"8010759921-6487407",
"8010759921-6487412",
"8010759921-21047221",
"8010759921-21047222",
"8010759941-6487408",
"8010759941-6487418",
"8010759951-6487409",
"8010759951-21047224",
8010759961,
8010760021,
8010760241,
8010760261,
8010760291,
8010760311,
8010760331,
8010764341,
8010764351,
8010764371,
8010764421,
8010764471,
8010764501,
8010764521,
8011318131,
8011321471,
8011325101,
8011325121,
8011325161,
8011325171,
8011325231,
8011330161,
8011330171,
8011330291,
8011330391,
8011330411,
8011330421,
8011330511,
8011330611,
8011330651,
8011330721,
8011332281,
8039851071,]

croppers = [str(i) for i in croppers]

for directs in croppers: 
    dir_in  ="W:\\BDADSharedData\\Spanish Flu\\Denmark\\Death Certificates 1916 - 1921\\"+directs
    dir_out = "W:\\BDADSharedData\\Spanish Flu\\Denmark\\Death Certificates 1916 - 1921\\"+directs+"_cropped"
# make dir_out if it doesn't exist yet ...
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    # loop through *.jpg files in the current direcotry and
    # call process_image on them
    for f in os.listdir(dir_in):
        if f[-4:] == ".jpg":         
            original = cv2.imread(dir_in+'\\'+f,1)
            crop     = autocrop(original)
            crop     = cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)          
            cv2.imwrite(os.path.join(dir_out, os.path.basename(f)), crop)
        

