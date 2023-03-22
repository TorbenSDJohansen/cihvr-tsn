# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Checks whether expected number of segmented images for every field.

"""


import os
import warnings

from typing import Dict

import pandas as pd


def count(root: str) -> Dict[str, int]:
    counts = {}
    cells = os.listdir(root)

    for i, cell in enumerate(cells, start=1):
        folder = os.path.join(root, cell)

        if not os.path.isdir(folder):
            continue

        print(f'Counting files in {folder} ({i}/{len(cells)})')
        files = os.listdir(folder)
        images = [x for x in files if x.endswith('.jpg')]

        if not len(files) == len(images):
            warnings.warn(f'Number files/folders in {folder} does not match number ending with ".jpg"')

        counts[cell] = len(images)

    return counts


def main():
    root_type_a = r'Y:\RegionH\Scripts\data\storage\minipics\TypeA'
    root_type_b = r'Y:\RegionH\Scripts\data\storage\minipics\TypeB'

    counts_type_a = count(root_type_a)
    counts_type_b = count(root_type_b)

    df_type_a = pd.DataFrame(list(counts_type_a.items()), columns=['Field', 'Count (A)'])
    df_type_b = pd.DataFrame(list(counts_type_b.items()), columns=['Field', 'Count (B)'])

    merged = df_type_a.merge(df_type_b, on='Field', how='outer')
    merged.to_csv(r'Y:\RegionH\Scripts\data\storage\minipics\counts.csv', index=False)


if __name__ == '__main__':
    main()
