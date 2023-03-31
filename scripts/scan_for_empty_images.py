# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:27:07 2023

@author: sa-tsdj
"""


import os

import pandas as pd


def main():
    root = r'Y:\RegionH\Scripts\data\storage\minipics\TypeA'
    subfolders = [x for x in os.listdir(root) if x.startswith('tab-b')]

    smallest_size_file = []

    for i, folder in enumerate(subfolders, start=1):
        print(f'Folder {i} of {len(subfolders)}: {folder}')
        folder = os.path.join(root, folder)
        files = [os.path.join(folder, x) for x in os.listdir(folder)]
        sizes = [os.path.getsize(file) for file in files]
        smallest_size_file.append([os.path.basename(folder), min(sizes)])

    results = pd.DataFrame(smallest_size_file, columns=['Field', 'Smallest size'])
