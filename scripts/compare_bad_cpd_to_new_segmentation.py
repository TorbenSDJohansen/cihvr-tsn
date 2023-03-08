# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Compare (predicted values of) bad CPD for birthweight to new segmentations.

"""


import os

import cv2
import imutils

import numpy as np
import pandas as pd


def main():
    new_segmentations = os.listdir(r'Y:\RegionH\Scripts\data\storage\minipics\TypeA_bottom_smaller_crop\weight-0-mo')
    new_journals = [x.split('.')[0] for x in new_segmentations]

    preds = pd.read_csv(r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\weight\base\preds.csv')
    preds['field'] = preds['filename_full'].transform(lambda x: os.path.basename(os.path.dirname(x)))
    preds['journal'] = preds['filename_full'].transform(lambda x: os.path.basename(x).split('.')[0])
    preds = preds[preds['field'] == 'weight-0-mo']
    preds = preds[preds['pred'] == 'bad cpd']

    overlap = sorted(set(new_journals) & set(preds['journal']))
    selected = np.random.choice(overlap, 20, replace=False)

    target_size = (209, 102)
    images = []

    for example in selected:
        old = cv2.imread(preds.loc[preds['journal'] == example, 'filename_full'].item())
        old = cv2.resize(old, target_size)

        new = os.path.join(r'Y:\RegionH\Scripts\data\storage\minipics\TypeA_bottom_smaller_crop\weight-0-mo', new_segmentations[new_journals.index(example)])
        new = cv2.imread(new)
        new = cv2.resize(new, target_size)

        images.extend([old, new])

    montage = imutils.build_montages(
        images,
        (2 * target_size[0] + 10, target_size[1] + 10),
        (2, 20),
        )[0]

    cv2.imwrite('./compare_bad_cpd_to_new_segmentation.png', montage)


if __name__ == '__main__':
    main()
