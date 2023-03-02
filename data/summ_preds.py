# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import sys
import os
import argparse
import datetime

import pandas as pd

from .format_preds_cihvr import drop_duplicates, _load_maps


def _get_cell_groups():
    default_measure_months = (1, 2, 3, 4, 6, 9, 12)

    def _get_group(name: str, times: tuple = default_measure_months):
        return [name.format(x) for x in times]

    table_b_cols = ( # ORDER is crucial here
        'Home economic status',
        'Home harmony',
        'Mother mental capacity',
        'Mother physical capacity',
        'Mother daily hours working at home',
        'Mother daily hours working outside home',
        'Nursery or kindergarten',
        'Care and cleanliness',
        'Own bed',
        'In air',
        'Smiles',
        'Lifts head',
        'Babbles',
        'Sits',
        'Nutrition',
        'Number of daily meals',
        )
    inv_mapping = { # NOTE: this works on raw names, not CIHVR names
        'Weight': _get_group('weight-{}-mo', (0, 1, 2, 3, 4, 6, 9, 12)),
        'Date': _get_group('date-{}-mo'),
        **{n: _get_group(f'tab-b-c{i}-{{}}-mo') for i, n in enumerate(table_b_cols, 1)},
        'Length': _get_group('length-{}-mo', (0, 12)),
        'Duration breastfeeding': ['dura-any-breastfeed'],
        'Nurse name': ['nurse-name-1', 'nurse-name-2', 'nurse-name-3'],
        }

    k = 0
    mapping = {}

    for key, value in inv_mapping.items():
        k += len(value)
        for subvalue in value:
            mapping[subvalue] = key

    assert k == len(mapping)

    return mapping


def _parse():
    parser = argparse.ArgumentParser(description='Summarize pred. files results.')

    # REQUIRED
    parser.add_argument(
        'files', type=str, nargs='+',
        help='The files of predictions to summarize.',
        )

    # OPTIONAL
    parser.add_argument(
        '--cihvr_duplicate_drop', default=False, action='store_true',
        help=(
            'Whether to perform journal duplicate drop based on bad cpd ' +
            'count. Important for CIHVR, but not in general.'
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

    args.files = [
        r'Z:\faellesmappe\tsdj\cihvr-new\dabf\eval-2021-05-18-14-30-12\eval-preds.csv',
        r'Z:\faellesmappe\tsdj\cihvr-new\date\eval-2021-05-18-14-30-48\eval-preds.csv',
        # r'Z:\faellesmappe\tsdj\cihvr-new\firstname\eval-2021-05-20-06-47-15\matched-eval-preds.csv',
        # r'Z:\faellesmappe\tsdj\cihvr-new\lastname\eval-2021-05-18-14-32-04\matched-eval-preds.csv',
        r'Z:\faellesmappe\tsdj\cihvr-new\length\eval-2021-05-18-14-32-54\eval-preds.csv',
        r'Z:\faellesmappe\tsdj\cihvr-new\tab_b\eval-2021-05-24-09-58-03\eval-preds.csv',
        r'Z:\faellesmappe\tsdj\cihvr-new\tab_b\eval-2021-05-24-10-02-39\eval-preds.csv',
        r'Z:\faellesmappe\tsdj\cihvr-new\weight\eval-2021-05-18-14-33-29\eval-preds.csv',
        ]
    args.cihvr_duplicate_drop = True
    args.threshold = 0.0
    # args.threshold = 0.5

    return args


def _format_args(args):
    print(f'Parsed args: {args}.')
    assert 0 <= args.threshold < 1
    for file in args.files:
        if not os.path.isfile(file):
            raise Exception(f'Non-existing file {file} requested!')

    return args.files, args.cihvr_duplicate_drop, args.threshold


def _create_summary_table(dataframe):
    return dataframe.groupby('meta_cell')['correct'].agg(['mean', 'count']).reset_index()


def main():
    """
    Summarize results of one or multiple prediction files. Provides results by
    cell, meta-group of cells, with/wihtout bad CPD, and when thresholding.

    NOTE: If multple datasets referring to same cell, no distinction between
    where the prediction comes from is made. Further, if multiple predictions
    on the same mini pic, these will be duplicate handled. In general, there
    should be NO reason to ever include multiple sets referring to same mini
    pic, because after all where should there ever be?

    Returns
    -------
    None.

    """
    print(
        'WARNING\n\n' +
        'Note a weakness of this script: If multiple predictions on same ' +
        'image, this is caught by duplicate drop and dropped, even though ' +
        'there might be cases where both are wanted, such as for names, ' +
        'in case seperate for first and last names!\n'
        )

    if len(sys.argv) > 1:
        args = _parse()
    else:
        args = _test_parse()

    files, cihvr_duplicate_drop, threshold = _format_args(args)

    print('Loading data!')
    _, map_images_journals_ss = _load_maps()
    pred_df = pd.concat([
        pd.read_csv(file, na_values=[''], keep_default_na=False) for file in files
        ])
    pred_df = pred_df[pred_df['prob'] >= threshold].copy()

    print('Creating new columns!')

    pred_df['cell'] = pred_df['filename_full'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    pred_df['filename'] = pred_df['filename_full'].apply(os.path.basename)
    pred_df['journal'] = pred_df['filename'].apply(lambda x: map_images_journals_ss[x])

    pred_df['colname_cihvr_data'] = pred_df['cell']

    pred_df['colname_pred'] = pred_df['colname_cihvr_data'].astype(str) + '_pred'
    pred_df['colname_prob'] = pred_df['colname_cihvr_data'].astype(str) + '_prob'

    if cihvr_duplicate_drop:
        print('Handling CIHVR duplicates!')
        pred_df = drop_duplicates(pred_df)

    # Group columns into meta-categories
    group_mapping = _get_cell_groups()
    pred_df['meta_cell'] = [group_mapping.get(x, x) for x in pred_df['cell'].values]

    print('Summarizing!')
    pred_df['correct'] = pred_df['pred'] == pred_df['label']

    by_cell = _create_summary_table(pred_df)
    by_cell_no_bad_cpd_label = _create_summary_table(pred_df[pred_df['label'] != 'bad cpd'])

    # Non-consistent empty coding is a bit messy
    by_cell_no_empty_label = _create_summary_table(
        pred_df[
            (pred_df['label'] != '0=Mangler')
            & (pred_df['label'] != ',:,')
            & (pred_df['label'] != 'empty')
            ]
        )
    by_cell_no_bad_cpd_or_empty_label = _create_summary_table(
        pred_df[
            (pred_df['label'] != '0=Mangler')
            & (pred_df['label'] != ',:,')
            & (pred_df['label'] != 'empty')
            & (pred_df['label'] != 'bad cpd')
            ]
        )

    # Merge together...
    results = by_cell.merge(by_cell_no_bad_cpd_label, on='meta_cell', how='left')
    results = results.merge(by_cell_no_empty_label, on='meta_cell', how='left')
    results = results.merge(by_cell_no_bad_cpd_or_empty_label, on='meta_cell', how='left')

    results.columns = [
        'Data',
        'Accuracy (all)', 'Count (all)',
        'Accuracy (successful crop)', 'Count (successful crop)',
        'Accuracy (non-empty)', 'Count (non-empty)',
        'Accuracy (successful crops and non-empty)', 'Count (successful crops and non-empty)',
        ]

    reordered_cols = list(results.columns[:1]) + sorted(results.columns[1:])
    results = results[reordered_cols]

    # Cast percentage and round
    results[results.columns[1:5]] = (100 * results[results.columns[1:5]]).round(1)

    # Write .tex

    with pd.option_context("max_colwidth", 1000):
        results_str = results[results.columns[:5]].to_latex(
            index=False,
            escape=False,
            )

    results_str = '\n'.join(results_str.split('\n')[2:-3])

    date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

    with open(f'Y:/RegionH/Scripts/users/tsdj/storage/results/transcr-accs-{date}.tex', 'w') as file:
        print(results_str, file=file)


if __name__ == '__main__':
    main()
