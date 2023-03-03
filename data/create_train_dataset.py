# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import argparse
import os
import shutil
import multiprocessing

from typing import Dict, List

import numpy as np
import pandas as pd


class MPCopier():
    def __init__(self, in_dir: str, out_dir: str, field: str):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.field = field

    def copy(self, file):
        in_file = os.path.join(self.in_dir, file)
        out_file = os.path.join(self.out_dir, '-'.join((self.field, file)))

        shutil.copyfile(in_file, out_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str)
    parser.add_argument('--labels-subdir', type=str, default='')
    parser.add_argument('--fields', type=str, nargs='+')
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--name', type=str, help='name of merged fields')
    parser.add_argument('--nb-pools', type=int, default=0)

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f'--dir {args.dir} does not exist')

    if not os.path.isdir(args.out_dir):
        raise NotADirectoryError(f'--out-dir {args.out_dir} does not exist')

    if args.nb_pools < 0:
        raise ValueError(f'cannot specify fewer than 0 pool, but got --nb-pools {args.nb_pools}')

    return args


def _load_labels(directory: str, fields: List[str]) -> pd.DataFrame:
    labels = []

    for field in fields:
        file = os.path.join(directory, '.'.join((field, 'npy')))

        if not os.path.isfile(file):
            raise FileNotFoundError(f'label file {field} does not exist')

        _labels = np.load(file, allow_pickle=True)
        _labels = pd.DataFrame(_labels, columns=['Filename', 'label'])
        _labels['field'] = field

        labels.append(_labels)

    labels = pd.concat(labels)

    return labels


def load_labels(directory: str, fields: List[str]) -> pd.DataFrame:
    labels = []

    for subdir in ('train', 'test'):
        full_dir = os.path.join(directory, subdir)

        if not os.path.isdir(full_dir):
            raise NotADirectoryError(f'label dir {full_dir} does not exist')

        _labels = _load_labels(full_dir, fields)
        _labels['split'] = subdir

        labels.append(_labels)

    labels = pd.concat(labels)

    # Change Filename to include the field in front to match new image name
    labels['Filename'] = labels['field'] + '-' + labels['Filename']

    return labels


def save_labels(labels: pd.DataFrame, out_dir: str, name: str):
    train = labels[labels['split'] == 'train']
    test = labels[labels['split'] == 'test']

    np.save(
        file=os.path.join(out_dir, 'labels', 'train', '.'.join((name, 'npy'))),
        arr=train[['Filename', 'label']].values,
        )
    np.save(
        file=os.path.join(out_dir, 'labels', 'test', '.'.join((name, 'npy'))),
        arr=test[['Filename', 'label']].values,
        )


def _move_images(
        labels: pd.DataFrame,
        image_folder: str,
        out_dir: str,
        field: str,
        nb_pools: int,
        ):
    image_files = set(os.listdir(image_folder))
    label_files = set(labels['Filename'])

    image_files = image_files & label_files
    # TODO do os.listdir(out_dir), then do setminus like: image_files -= set(os.listdir(out_dir))
    # NOTE: This is slow, can move os.listdir() to move_images_by_copy and re-use, that is better way
    # NOTE: Could potentially remove from labels in move_images_by_copy by using the os.listdir, ~['Filename'].isin(...)

    print(f'copying {len(image_files)} from {image_folder}')

    copier = MPCopier(in_dir=image_folder, out_dir=out_dir, field=field)

    if nb_pools > 0:
        with multiprocessing.Pool(nb_pools) as pool:
            pool.map(copier.copy, image_files)
    else:
        for file in image_files:
            copier.copy(file)


def move_images_by_copy(
        labels: pd.DataFrame,
        image_folders: Dict[str, str],
        fields: List[str],
        out_dir: str,
        nb_pools: int,
        ):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=False)

    for field in fields: # TODO count report count, track = 0, track += len(sub), divide by len(labels)
        sub = labels[labels['field'] == field]
        image_folder = image_folders[field]

        print(f'copying images for {field} to {out_dir}')

        _move_images(sub, image_folder, out_dir, field, nb_pools)


def main():
    args = parse_args()

    image_dir = os.path.join(args.dir, 'minipics')
    image_folders = {x: os.path.join(image_dir, x) for x in os.listdir(image_dir)}

    if not set(args.fields).issubset(image_folders.keys()):
        unmatched = set(args.fields) - set(image_folders.keys())
        raise NotADirectoryError(f'at least one of fields {args.fields} does not have a corresponding image folder in {image_dir}: {unmatched}')

    label_dir = os.path.join(args.dir, 'labels', args.labels_subdir)
    labels = load_labels(label_dir, args.fields)

    save_labels(labels=labels, out_dir=args.out_dir, name=args.name)

    move_images_by_copy(
        labels=labels,
        image_folders=image_folders,
        fields=args.fields,
        out_dir=os.path.join(args.out_dir, 'minipics', args.name),
        nb_pools=args.nb_pools,
        )


if __name__ == '__main__':
    main()
