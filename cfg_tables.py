# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import os
import argparse

from typing import Any, Dict, List, Union

import yaml

import pandas as pd

from timmsn.data.formatters import create_formatter

import argparser # needed to get formatters -> seq. len. pylint: disable=W0611


NUM_GPUS = {
    r'date\mn': 2,
    r'first\mh-tl': 2,
    r'weight\mh': 2,
    r'circle\s2s': 1,
    r'int\s2s-5d': 2,
    r'last\s2s-tl': 2,
    }

def parse_args():
    default = [
        r'Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\circle\s2s\args.yaml',
        r'Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml',
        r'Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\args.yaml',
        r'Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s-tl\args.yaml',
        r'Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl\args.yaml',
        r'Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\mh\args.yaml',
        ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--files', type=str, nargs='+', default=default)
    parser.add_argument('--out-dir', type=str, default='')

    args = parser.parse_args()

    if len(args.files) != len(set(args.files)):
        raise ValueError('same arg file specified multiple times: {arg_files}')

    return args


def load(fname: str) -> dict:
    with open(fname, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def create_colnames(arg_files, current_pos=-3):
    colnames = [os.path.sep.join(x.split(os.path.sep)[current_pos:-1]) for x in arg_files]

    if not len(colnames) == len(set(colnames)):
        return create_colnames(arg_files, current_pos=current_pos - 1)

    return colnames


def create_diff_ds(
        cfgs: List[Dict[str, Any]],
        colnames: List[str],
        ) -> pd.DataFrame:
    keyset = set()

    for cfg in cfgs:
        keyset = keyset.union(cfg.keys())

    cfg_ds = {}

    for key in keyset:
        cfg_ds[key] = [cfg.get(key, None) for cfg in cfgs]

    cfg_ds = pd.DataFrame(cfg_ds).T
    cfg_ds.columns = colnames

    # Find args where there is at least one difference
    cfg_ds['diff'] = False

    for col in colnames[1:]:
        new_mask = (cfg_ds[colnames[0]] != cfg_ds[col]) & ~(cfg_ds[colnames[0]].isnull() & cfg_ds[col].isnull())
        cfg_ds['diff'] = cfg_ds['diff'] | new_mask

    diff = cfg_ds[cfg_ds['diff']].drop(columns='diff').sort_index()

    # Rows "experiment" and "output" not informative as already in column names
    diff = diff[~diff.index.isin(['experiment', 'output'])]

    return diff


def write_tex(cfg_ds: pd.DataFrame, fname: Union[str, os.PathLike]):
    # Prettier names
    cfg_ds = cfg_ds.rename(index={
        'dataset_cells': 'Fields',
        'input_size': 'Image size',
        'formatter': 'Tokenization',
        'batch_size': 'Batch size',

        })
    cfg_ds = cfg_ds.T

    # List values to str format
    # cfg_ds['Fields'] = cfg_ds['Fields'].transform(lambda x: ', '.join(x))
    cfg_ds['Fields'] = cfg_ds['Fields'].transform(', '.join)
    cfg_ds['Image size'] = cfg_ds['Image size'].transform(lambda x: 'x'.join([str(e) for e in x]))

    # Extract seq. len.
    formatters = [create_formatter(x) for x in cfg_ds['Tokenization']]
    cfg_ds['Seq. length'] = [len(x.num_classes) for x in formatters]

    # Drop some columns
    to_drop = [
         'Tokenization',
         'initial_log',
         'tl_from_input_size',
         'initial_checkpoint',
         ]
    to_drop = [x for x in to_drop if x in cfg_ds.columns]
    cfg_ds = cfg_ds.drop(columns=to_drop)

    # Select (or add as empty) columns in right order
    target_cols = [
        'Fields',
        'Image size',
        'Seq. length',
        'Batch size',
        ]

    for col in [x for x in target_cols if x not in cfg_ds.columns]:
        cfg_ds[col] = ''

    # Sort columns
    cfg_ds = cfg_ds[target_cols]

    # Fix batch size for multi-gpu models
    for model, num_gpus in NUM_GPUS.items():
        cfg_ds.loc[cfg_ds.index == model, 'Batch size'] *= num_gpus

    # Prettier index name
    cfg_ds.index = [os.path.dirname(x).capitalize() for x in cfg_ds.index]

    # Write .tex
    with pd.option_context("max_colwidth", 1000):
        table = cfg_ds.to_latex(
            index=True,
            escape=False,
            )

    cfg_ds_str = '\n'.join(table.split('\n')[4:-3])

    with open(fname, 'w', encoding='utf-8') as file:
        print(cfg_ds_str, file=file)


def main():
    args = parse_args()

    all_cfgs = [load(x) for x in args.files]

    mh_cfgs = []
    mh_files = []

    s2s_cfgs = []
    s2s_files = []

    for file, cfg in zip(args.files, all_cfgs):
        if cfg['model'].lower().startswith('deit3'):
            s2s_files.append(file)
            s2s_cfgs.append(cfg)
        elif cfg['model'].lower().startswith('tf_efficientnetv2'):
            mh_files.append(file)
            mh_cfgs.append(cfg)
        else:
            raise ValueError(f'Unexpected model type {cfg["model"]}')

    mh_ds = create_diff_ds(mh_cfgs, create_colnames(mh_files))
    s2s_ds = create_diff_ds(s2s_cfgs, create_colnames(s2s_files))

    write_tex(
        cfg_ds=mh_ds,
        fname=os.path.join(args.out_dir, 'mh-models-diff.tex'),
        )
    write_tex(
        cfg_ds=s2s_ds,
        fname=os.path.join(args.out_dir, 's2s-models-diff.tex'),
        )


if __name__ == '__main__':
    main()
