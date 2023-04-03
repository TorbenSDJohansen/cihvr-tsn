# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Attempt to identify if score obtained during segmentation can be used to obtain
cases of "bad segmentation", which in turn can be used to either:
    (1) Add by-field labels of such cases
    (2) Remove cases on a by-full-image level

"""


import os
import pickle
import shutil

from typing import Union

import pandas as pd


def load() -> pd.DataFrame:
    directory = r'Y:\RegionH\Scripts\data\storage\tableparsing_summary\TypeA'
    data = []

    for fname in os.listdir(directory):
        fname = os.path.join(directory, fname)

        with open(fname, 'rb') as file:
            data.extend(pickle.load(file))

    data = pd.DataFrame(data, columns=['file', 'score'])
    data = data[data['score'] != 'Not TypeA']
    data['Journal'] = data['file'].transform(lambda x: os.path.basename(x).split('.')[0])
    data = data.drop(columns='file')

    return data


def subset(
        data: pd.DataFrame,
        folder_a: Union[str, os.PathLike],
        folder_b: Union[str, os.PathLike],
        field: str,
        ) -> pd.DataFrame:
    candidates = [x.split('.')[0] for x in os.listdir(os.path.join(folder_a, field))]
    exclude = {
        *[x.split('.')[0] for x in os.listdir(os.path.join(folder_b, 'district-3'))], # top half
        *[x.split('.')[0] for x in os.listdir(os.path.join(folder_b, 'weight-1-mo'))], # bot half
        }

    # Include only those where
    sub = data[data['Journal'].isin(candidates)]

    # Exclude those of Type B
    # ends up not removing anything, caught by `data[data['score'] != 'Not TypeA']`
    sub = sub[~sub['Journal'].isin(exclude)]

    return sub


def main():
    data = load()

    folder_a = r'Y:\RegionH\Scripts\data\storage\minipics\TypeA'
    folder_b = r'Y:\RegionH\Scripts\data\storage\minipics\TypeB'
    field = 'district-3'
    nb_to_check: int = 200
    out_dir: str = './tmp-output/'

    sub = subset(
        data=data,
        folder_a=folder_a,
        folder_b=folder_b,
        field=field,
        )
    sub_sored = sub.sort_values('score').reset_index(drop=True)

    for i in range(nb_to_check):
        in_file = os.path.join(folder_a, field, sub_sored.loc[i, 'Journal'] + '.jpg')

        # Useful to prepend score to sort images in folder
        score = sub_sored.loc[i, 'score']
        score = str(score)[2:6]
        out_file = os.path.join(out_dir, score + '-' + os.path.basename(in_file))

        shutil.copy(in_file, out_file)

    # Move to script or data
    # If useful to obtain "bad segmentation", place appropriately in README and
    # use in data/gen_labels.py


if __name__ == '__main__':
    main()
