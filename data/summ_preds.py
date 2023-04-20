# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import os
import argparse
import datetime
import warnings

import pandas as pd

from format_preds_cihvr import (
    drop_duplicates,
    load_maps,
    get_cell_groups,
    derive_cell,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize pred. files results.')

    # REQUIRED
    parser.add_argument(
        'files', type=str, nargs='+',
        help='The files of predictions to summarize.',
        )
    parser.add_argument('--out-dir', type=str)

    # OPTIONAL
    parser.add_argument(
        '--cihvr-duplicate-drop', default=False, action='store_true',
        help=(
            'Whether to perform journal duplicate drop based on bad cpd ' +
            'count. Important for CIHVR, but not in general.'
            ),
        )
    parser.add_argument(
        '--threshold', type=float, default=0,
        help='The prob. thresholding applied, droppings predictions below value.',
        )
    parser.add_argument('--no-field-merge', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        raise NotADirectoryError(f'--out-dir {args.out_dir} does not exist')

    return args


def format_args(args):
    print(f'Parsed args: {args}.')
    assert 0 <= args.threshold < 1
    for file in args.files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f'Non-existing file {file} requested!')

    return args.files, args.cihvr_duplicate_drop, args.threshold


def merge_on_cpr_info(data: pd.DataFrame) -> pd.DataFrame:
    cpr = pd.read_stata('Y:/RegionH/SPJ/CPR info and duplicate info/spj_cprlist-plusID_180227_181106.dta')

    cpr['dob'] = cpr['cpr'].apply(lambda x: x[:-4])
    cpr['bdy'] = cpr['dob'].apply(lambda x: x[4:])

    cpr = cpr.rename(columns={'id_s': 'Id'})
    cpr = cpr.drop(columns=['id_c', 'dob', 'cpr'])

    data = data.merge(cpr, on='Id', how='left')

    return data


def merge_on_birth_info(pred_df) -> pd.DataFrame:
    main_df = pd.read_csv('Y:/RegionH/SPJ/Database/export_181106.txt', sep=';', na_values=[''], keep_default_na=False)

    main_df = main_df[['Id', 'Filename']]
    main_df = main_df.rename(columns={'Filename': 'journal'})
    main_df = merge_on_cpr_info(main_df)
    main_df = main_df.drop(columns='Id')

    pred_df = pred_df.merge(main_df, on='journal', how='left')

    return pred_df


def _create_summary_table(dataframe):
    return dataframe.groupby('meta_cell')['correct'].agg(['mean', 'count']).reset_index()


def create_table_by_birthyear(data: pd.DataFrame, fname: str):
    birth_years = [str(x) for x in range(59, 68)]
    results = None
    colnames = ['Data']

    for bdy in birth_years:
        by_cell_x_bdy = _create_summary_table(data[data['bdy'] == bdy])

        if results is None:
            results = by_cell_x_bdy.copy()
        else:
            results = results.merge(by_cell_x_bdy, on='meta_cell', how='inner')

        colnames.extend([f'Accuracy ({bdy})', f'Count ({bdy})'])

    results.columns = colnames

    reordered_cols = list(results.columns[:1]) + sorted(results.columns[1:])
    results = results[reordered_cols]

    # Cast percentage and round
    results[results.columns[1:len(birth_years) + 1]] = (100 * results[results.columns[1:len(birth_years) + 1]]).round(1)

    # Write .tex

    with pd.option_context("max_colwidth", 1000):
        results_str = results[results.columns[:len(birth_years) + 1]].to_latex(
            index=False,
            escape=False,
            )

    results_str = '\n'.join(results_str.split('\n')[4:-3])

    with open(fname, 'w', encoding='utf-8') as file:
        print(results_str, file=file)


def create_table_by_subset(
        pred_df,
        no_bad_cpd_label,
        no_empty_label,
        no_bad_cpd_or_empty_label,
        fname: str,
        ):
    # Tables
    by_cell = _create_summary_table(pred_df)
    by_cell_no_bad_cpd_label = _create_summary_table(no_bad_cpd_label)
    by_cell_no_empty_label = _create_summary_table(no_empty_label)
    by_cell_no_bad_cpd_or_empty_label = _create_summary_table(no_bad_cpd_or_empty_label)

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

    results_str = '\n'.join(results_str.split('\n')[4:-3])

    with open(fname, 'w', encoding='utf-8') as file:
        print(results_str, file=file)


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
    warnings.warn(
        'Note a weakness of this script: If multiple predictions on same ' +
        'image, this is caught by duplicate drop and dropped, even though ' +
        'there might be cases where both are wanted, such as for names, ' +
        'in case seperate for first and last names!\n'
        )

    args = parse_args()
    files, cihvr_duplicate_drop, threshold = format_args(args)

    print('Loading data!')
    _, map_images_journals_ss = load_maps()
    pred_df = pd.concat([
        pd.read_csv(file, na_values=[''], keep_default_na=False) for file in files
        ])
    pred_df = pred_df[pred_df['prob'] >= threshold].copy()

    group_mapping = get_cell_groups()

    print('Creating new columns!')
    pred_df['filename'] = pred_df['filename_full'].apply(os.path.basename)
    pred_df['cell'] = pred_df['filename_full'].apply(lambda x: derive_cell(x, list(group_mapping.keys())))

    # When filename is, e.g., breastfeed-7-do-SP2_19239.pdf.page-0.jpg, need to
    # change to SP2_19239.pdf.page-0.jpg
    for cell in pred_df['cell'].unique():
        to_change = (pred_df['cell'] == cell) & pred_df['filename'].str.startswith(cell)
        pred_df.loc[to_change, 'filename'] = pred_df.loc[to_change, 'filename'].apply(lambda x: x[(len(cell) + 1):])

    pred_df['journal'] = pred_df['filename'].apply(lambda x: map_images_journals_ss[x])
    pred_df = merge_on_birth_info(pred_df)

    pred_df['colname_cihvr_data'] = pred_df['cell']

    pred_df['colname_pred'] = pred_df['colname_cihvr_data'].astype(str) + '_pred'
    pred_df['colname_prob'] = pred_df['colname_cihvr_data'].astype(str) + '_prob'

    if cihvr_duplicate_drop:
        print('Handling CIHVR duplicates!')
        pred_df = drop_duplicates(pred_df)

    if args.no_field_merge:
        pred_df['meta_cell'] = pred_df['cell']
    else:
        # Group columns into meta-categories
        pred_df['meta_cell'] = [group_mapping.get(x, x) for x in pred_df['cell'].values]

    print('Summarizing!')
    pred_df['correct'] = pred_df['pred'] == pred_df['label']

    # Subsets
    no_bad_cpd_label = pred_df[pred_df['label'] != 'bad cpd']
    no_empty_label = pred_df[ # Non-consistent empty coding is a bit messy
        (pred_df['label'] != '0=Mangler')
        & (pred_df['label'] != ',:,')
        & (pred_df['label'] != 'empty')
        ]
    no_bad_cpd_or_empty_label = pred_df[
        (pred_df['label'] != '0=Mangler')
        & (pred_df['label'] != ',:,')
        & (pred_df['label'] != 'empty')
        & (pred_df['label'] != 'bad cpd')
        ]

    # File names
    date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    fn_split = os.path.join(args.out_dir, f'transcr-accs-split-{date}.tex')
    fn_years = os.path.join(args.out_dir, f'transcr-accs-years-{date}.tex')
    fn_years_no_empty = os.path.join(args.out_dir, f'transcr-accs-years-no-empty-{date}.tex')

    # Tables
    create_table_by_subset(
        pred_df=pred_df,
        no_bad_cpd_label=no_bad_cpd_label,
        no_empty_label=no_empty_label,
        no_bad_cpd_or_empty_label=no_bad_cpd_or_empty_label,
        fname=fn_split,
        )
    create_table_by_birthyear(pred_df, fn_years)
    create_table_by_birthyear(no_empty_label, fn_years_no_empty)


if __name__ == '__main__':
    main()
