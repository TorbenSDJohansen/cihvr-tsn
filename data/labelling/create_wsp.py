# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Current training sets are heavily selected for some fields with few and in many
cases even zero examples of empty fields.

This scripts creates workspaces from prediction .csv on a per-field basis,
while making sure to not add any which are already in our current labels. It is
possible to sample "empty", "random", or "low-prob".

Example use: Use a first round of predictions to select likely empty and then
create work-space for labelling.

"""


import argparse
import math
import os
import warnings

from typing import List, Union

import json

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--fn-preds', type=str)
    parser.add_argument('--label-dir', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('-n', type=int, default=100, help='number selected per field')
    parser.add_argument('--package-size', type=int, default=None, help='maximum package size')
    parser.add_argument('--sample', type=str, default='empty', choices=['empty', 'random', 'low-prob'])

    args = parser.parse_args()

    if not os.path.isfile(args.fn_preds):
        raise FileNotFoundError(f'--fn-preds {args.fn_preds} does not exist')

    if not os.path.isdir(args.label_dir):
        raise NotADirectoryError(f'--label-dir {args.label_dir} does not exist')

    if not os.path.isdir(args.outdir):
        raise NotADirectoryError(f'--outdir {args.outdir} does not exist')

    if args.n <= 0:
        raise ValueError('-n must be greater than 0')

    if args.package_size is not None and args.package_size <= 100 :
        raise ValueError('--package-size must be greater than 100')

    return args


def create_workspaces(
        to_label: pd.DataFrame,
        package_size: int,
        outdir: Union[str, os.PathLike],
        ):
    wsp = {
        'tags': ['field'],
        'currentTag': 'field',
        'isModified': False,
        'savePath': None,
        'cursor': 0,
        'useInference': True,
        }

    package_size = package_size if package_size else len(to_label)
    nb_packages = math.ceil(len(to_label) / package_size)
    offset = math.ceil(len(to_label) / nb_packages)
    to_label = to_label[['filename_full', 'pred']]

    for i in range(nb_packages):
        elements = []

        for filename, pred in to_label.values[i * offset: (i + 1) * offset]:
            dirname, basename = os.path.split(filename)

            elements.append({
                'name': basename,
                'folder': dirname,
                'path': filename,
                'properties': {'field': pred},
                })

        wsp['elements'] = elements # re-use and overwrite this part

        fn_out = os.path.join(outdir, f'wsp-{i}.json')

        if os.path.isfile(fn_out):
            warnings.warn(f'wsp file {fn_out} already exists, skipping')
            continue

        with open(fn_out, 'w', encoding='utf-8') as file:
            json.dump(wsp, file)


def get_current_labels(
        label_dir: Union[str, os.PathLike],
        fields: List[str],
        ) -> pd.DataFrame:
    subdirs = os.listdir(label_dir)
    labels = []

    for subdir in subdirs:
        path = os.path.join(label_dir, subdir)

        if not os.path.isdir(path):
            raise NotADirectoryError(f'requested label subdir {path} is not a dir')

        for field in fields:
            for split in ('train', 'test'):
                fname = os.path.join(path, split, '.'.join([field, 'npy']))

                if not os.path.isfile(fname):
                    continue

                tmp = np.load(fname, allow_pickle=True)
                tmp = pd.DataFrame(tmp, columns=['image', 'label'])
                tmp['field'] = field
                tmp['subdir'] = subdir

                labels.append(tmp)

    labels = pd.concat(labels)

    assert (labels['image'] + labels['field'] + labels['subdir']).nunique() == len(labels)

    return labels


def select_images_empty(sub: pd.DataFrame, number: int) -> pd.DataFrame:
    empty = sub[sub['pred'] == '0=Mangler']

    if len(empty) < number: # add those predicted with lowest certainty
        diff = number - len(empty)
        complement = sub[sub['pred'] != '0=Mangler']
        complement = complement.nsmallest(diff, columns='prob')
        empty = pd.concat([empty, complement])
    else:
        empty = empty.sample(n=number, replace=False, random_state=42)

    return empty


def select_images_random(sub: pd.DataFrame, number: int) -> pd.DataFrame:
    return sub.sample(n=number, replace=False, random_state=42)


def select_images_low_prob(sub: pd.DataFrame, number: int) -> pd.DataFrame:
    return sub.nsmallest(number, columns='prob')


def main():
    args = parse_args()

    preds = pd.read_csv(args.fn_preds)
    preds['field'] = preds['filename_full'].transform(lambda x: os.path.basename(os.path.dirname(x)))
    preds['image'] = preds['filename_full'].transform(os.path.basename)
    preds['pred-empty'] = preds['pred'] == '0=Mangler'

    # Get current labels to make sure we don't select those
    labels = get_current_labels(args.label_dir, preds['field'].unique())

    to_label = []

    for field in preds['field'].unique():
        # Select specific field
        sub = preds[preds['field'] == field]

        # Make sure to not select image-field pairs already labelled
        to_exclude = labels.loc[labels['field'] == field, 'image']
        sub = sub[~sub['image'].isin(to_exclude)]

        if args.sample == 'empty':
            # Select those predicted as empty (or with low prob if not enough empty)
            sub = select_images_empty(sub, number=args.n)
        elif args.sample == 'random':
            sub = select_images_random(sub, number=args.n)
        elif args.sample == 'low-prob':
            sub = select_images_low_prob(sub, number=args.n)
        else:
            raise ValueError(f'--sample arg {args.sample} not allowed')

        to_label.append(sub)

    to_label = pd.concat(to_label)

    # Since "0=Mangler" long to type, replace with "" and map back when going
    # from workspace to labels
    to_label['pred'] = to_label['pred'].replace(to_replace='0=Mangler', value='')

    create_workspaces(
        to_label=to_label,
        package_size=args.package_size,
        outdir=args.outdir,
        )


if __name__ == '__main__':
    main()
