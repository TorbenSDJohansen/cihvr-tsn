# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import os

import pandas as pd


def get_name(fname: str) -> str:
    return os.path.basename(os.path.dirname(os.path.dirname(fname)))


def transform_f(filename: str) -> str:
    return os.path.basename(filename)


def load(fname: str) -> pd.DataFrame:
    pred = pd.read_csv(fname)
    pred = pred[~pred['label'].str.contains('0=Mangler')]

    pred['c'] = pred['label'] == pred['pred']

    pred = pred.rename(columns={'filename_full': 'f'})
    pred['f'] = pred['f'].transform(transform_f)

    name = get_name(fname)
    pred = pred.rename(columns={
        'label': '-'.join([name, 'label']),
        'pred': '-'.join([name, 'pred']),
        'prob': '-'.join([name, 'prob']),
        'c': '-'.join([name, 'c']),
        })

    return pred


def main():
    first = load(r'Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\s2s-tl\preds.csv')
    last = load(r'Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\s2s-tl\preds.csv')
    both = load(r'Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first-and-last\s2s-tl\preds.csv')

    merged = first.merge(last, on='f', how='inner').merge(both, on='f', how='inner')

    merged['nn-post-join'] = merged['first-pred'] + ' ' + merged['last-pred']
    merged['c-post-join'] = merged['nn-post-join'] == merged['first-and-last-label']

    print(merged['c-post-join'].mean())
    print(merged['first-and-last-c'].mean())

    # Check labels consisten between joint and first + last name models
    merged['label-post-join'] = merged['first-label'] + ' ' + merged['last-label']
    assert (merged['label-post-join'] == merged['first-and-last-label']).all()


if __name__ == '__main__':
    main()
