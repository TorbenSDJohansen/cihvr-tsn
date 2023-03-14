# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:44:01 2023

@author: sa-cmd
"""

import glob
import os
import shutil
import tqdm


# Basedir
basedir = 'Y:/RegionH/Scripts/data/storage/minipics'
# Source path 
source_folder = 'Y:/RegionH/Scripts/data/storage/minipics/TypeA_bottom_smaller_crop'
  
# Destination path 
destination_folder = os.path.abspath('Y:/RegionH/Scripts/data/storage/minipics/TypeA/')

# Search files with .txt extension in source directory
pattern = "/**/*.jpg"
files = glob.glob(source_folder + pattern)

# move the files with txt extension
for file in tqdm.tqdm(files):
    # extract file name form file path    
    shutil.move(file, os.path.join(destination_folder,os.path.relpath(file, start=source_folder)))

   

# compressed_file = shutil.make_archive(
#         base_name='archive',   # archive file name w/o extension
#         format='zip',        # available formats: zip, gztar, bztar, xztar, tar
#         root_dir='path/to/dir' # directory to compress
# )

