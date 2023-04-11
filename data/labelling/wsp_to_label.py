# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Take workspaces created by create_wsp.py (and then manually reviwed) and create
label file(s).

"""


import argparse
import os
import pickle

import json

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--wsp-dir', type=str)
    parser.add_argument('--fn-out', type=str)

    args = parser.parse_args()

    if not os.path.isdir(args.wsp_dir):
        raise NotADirectoryError(f'--wsp-dir {args.wsp_dir} does not exist')

    if os.path.isfile(args.fn_out):
        raise FileExistsError(f'--fn-out {args.fn_out} already exist')

    if not os.path.isdir(os.path.dirname(args.fn_out)):
        raise NotADirectoryError(f'--fn-out {args.fn_out} directory does not exist')

    return args


def load_workspace(file: str) -> pd.DataFrame:
    with open(file, 'r', encoding='utf-8') as stream:
        workspace = json.load(stream)

    cursor = workspace['cursor'] + 1 # image reached in workspace
    workspace = workspace['elements']

    if cursor != len(workspace):
        raise ValueError(f'workspace {file} not appear to gone all trough, cursor = {cursor} != {len(workspace)} = len(workspace)')

    workspace = workspace[:cursor] # only keep images reached

    labels = pd.DataFrame({
        'image_id':[x['name'] for x in workspace],
        'field': [os.path.basename(x['folder']) for x in workspace],
        'label': [x['properties']['field'] for x in workspace],
        })

    return labels


def workspace_to_labels_dataframe(wsp_dir: dir) -> pd.DataFrame:
    wsp_files = [os.path.join(wsp_dir, x) for x in os.listdir(wsp_dir)]
    wsp_files = [x for x in wsp_files if x.endswith('.json')]

    if len(wsp_files) == 0:
        raise FileNotFoundError(f'no workspaces found in {wsp_dir}')

    print(f'creating labels from {len(wsp_files)} workspaces')

    labels = []

    for file in wsp_files:
        labels.append(load_workspace(file))

    labels = pd.concat(labels)

    # Only keep labels with values we expect/are useable
    allowed_values = {'', *{str(x) for x in range(10_000)}}
    new_labels = labels[labels['label'].isin(allowed_values)].copy()

    # Change '' to '0=Mangler' format for empty
    new_labels['label'] = new_labels['label'].replace('', '0=Mangler')

    return new_labels


def main():
    args = parse_args()

    new_labels = workspace_to_labels_dataframe(args.wsp_dir)

    fn_map_lookup_df = 'Y:/RegionH/Scripts/users/tsdj/storage/maps/map_lookup_df.pkl'

    with open(fn_map_lookup_df, 'rb') as file:
        map_lookup_df = pickle.load(file)

    new_labels['field'] = new_labels['field'].replace(map_lookup_df)

    new_labels = new_labels.pivot(
        index='image_id',
        columns='field',
        values='label',
        ).reset_index()

    new_labels.to_csv(args.fn_out, index=False)


if __name__ == '__main__':
    main()
