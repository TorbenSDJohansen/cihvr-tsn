# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import os
import argparse
import pickle

import pandas as pd


def _load_maps():
    root_maps = 'Y:/RegionH/Scripts/users/tsdj/storage/maps'

    with open(f'{root_maps}/map_lookup_df.pkl', 'rb') as file:
        map_lookup_df = pickle.load(file)

    with open(f'{root_maps}/map_journals_images_ss.pkl', 'rb') as file:
        map_journals_images_ss = pickle.load(file)

    map_images_journals_ss = {}

    for key, value in map_journals_images_ss.items():
        for fname in value:
            map_images_journals_ss[fname] = key

    return map_lookup_df, map_images_journals_ss


def _parse():
    parser = argparse.ArgumentParser(description='Format pred. file to CIHVR format.')

    # REQUIRED
    parser.add_argument(
        'file', type=str,
        help='The file of predictions to format.',
        )

    # OPTIONAL
    parser.add_argument(
        '--to', type=str, default='wide', choices=['wide', 'long', 'both'],
        help='',
        )
    parser.add_argument(
        '--use_cihvr_name_if_available', default=False, action='store_true',
        help=(
            'Uses the associated original CIHVR name of the variable when'
            + ' available, otherwise keep name.'
            ),
        )
    parser.add_argument(
        '--threshold', type=float, default=0,
        help='The prob. thresholding applied, droppings predictions below value.',
        )

    args = parser.parse_args()

    return args


def _test_parse():
    parser = argparse.ArgumentParser(description='Format pred. file to CIHVR format.')
    args = parser.parse_args()

    args.file = r'Z:\faellesmappe\tsdj\cihvr-new\weight\preds-2021-05-19-11-14-24.csv' # pylint: disable=C0301
    args.to = 'both'
    args.use_cihvr_name_if_available = True
    args.threshold = 0

    return args


def _format_args(args):
    print(f'Parsed args: {args}.')
    assert os.path.isfile(args.file)
    assert 0 <= args.threshold < 1

    return args.file, args.to, args.use_cihvr_name_if_available


def drop_duplicates(pred_df: pd.DataFrame) -> pd.DataFrame:
    ''' Pivot does not work since duplicate journal entry when multiple pages
    of same journal cropped. However, also only needs to maintain ONE. As
    such, keep the "best" version - sorting away the one with the highest
    number of "bad cpd" cases.
    '''

    bad_cpd_count = pred_df.groupby('filename').apply(
        lambda x: (x['pred'] == 'bad cpd').sum()
        ).reset_index()
    bad_cpd_count.columns = list(bad_cpd_count.columns[:-1]) + ['bad-cpd-count']

    pred_df = pred_df.merge(bad_cpd_count, how='outer', on='filename')
    pred_df['is-min-bad-cpd-count'] = pred_df.groupby('journal')['bad-cpd-count'].transform(
        lambda x: x == min(x)
        )
    pred_df = pred_df[pred_df['is-min-bad-cpd-count']]

    # Ties are possible, which will break program. If they exist, keep the first
    # one ("arbitrarity" - but this is OK, no meaningful way to choose).
    pred_df['cumcount'] = pred_df.groupby(['journal', 'cell'])['pred'].cumcount()
    pred_df = pred_df[pred_df['cumcount'] == 0]

    pred_df = pred_df.drop(columns=['bad-cpd-count', 'is-min-bad-cpd-count', 'cumcount'])

    return pred_df


def main():
    """
    Reformat a prediction .csv to "CIHVR" format, either as long, wide, or
    both.

    Returns
    -------
    None.

    """
    args = _parse()
    # args = _test_parse()
    file, format_to, use_cihvr_name_if_available = _format_args(args)

    print('Loading data!')
    map_lookup_df, map_images_journals_ss = _load_maps()
    pred_df = pd.read_csv(file, na_values=[''], keep_default_na=False)

    pred_df = pred_df[pred_df['prob'] >= args.threshold]

    print('Creating new/renaming columns!')
    pred_df['cell'] = pred_df['filename_full'].str.split('/').apply(lambda x: x[-2])
    pred_df['filename'] = pred_df['filename_full'].str.split('/').apply(lambda x: x[-1])
    pred_df['journal'] = pred_df['filename'].apply(lambda x: map_images_journals_ss[x])

    if use_cihvr_name_if_available:
        pred_df['colname_cihvr_data'] = pred_df['cell'].apply(
            lambda x: map_lookup_df.get(x, x),
            )
    else:
        pred_df['colname_cihvr_data'] = pred_df['cell']

    pred_df['colname_pred'] = pred_df['colname_cihvr_data'].astype(str) + '_pred'
    pred_df['colname_prob'] = pred_df['colname_cihvr_data'].astype(str) + '_prob'

    print('Handling duplicates!')
    pred_df = drop_duplicates(pred_df)

    print('Reshaping data!')
    pred_df_wide_pred = pred_df.pivot(
        index='journal',
        columns='colname_pred',
        values='pred',
        ).reset_index()
    pred_df_wide_prob = pred_df.pivot(
        index='journal',
        columns='colname_prob',
        values='prob',
        ).reset_index()
    pred_df_wide = pred_df_wide_pred.merge(
        pred_df_wide_prob,
        on='journal',
        )
    assert len(pred_df_wide_pred) == len(pred_df_wide_prob) == len(pred_df_wide)

    print('Writing file(s)!')
    path, fname = os.path.dirname(file), os.path.basename(file)

    if format_to in ('long', 'both'):
        file_long = ''.join((path, '/long-', fname))
        if os.path.isfile(file_long):
            print(f'WARNING: Long file already exists: "{file_long}". Not writing!')
        else:
            print(f'Writing: "{file_long}."')
            pred_df.to_csv(file_long, index=False)

    if format_to in ('wide', 'both'):
        file_wide = ''.join((path, '/wide-', fname))
        if os.path.isfile(file_wide):
            print(f'WARNING: Wide file already exists: "{file_wide}". Not writing!')
        else:
            print(f'Writing: "{file_wide}."')
            pred_df_wide.to_csv(file_wide, index=False)


if __name__ == '__main__':
    main()
