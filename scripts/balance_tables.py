# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Create balance tables by born 1-3 to assess if some selection issue before up-
loading data to DST.

"""


import argparse
import os
import pickle

from typing import List, Tuple, Union

import scipy

import numpy as np
import pandas as pd

from compare_uploads import rem_decimal_if_possible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--fn-in', type=str)
    parser.add_argument('--fields', type=str, nargs='+', default=None)
    parser.add_argument('--fn-out', type=str)

    args = parser.parse_args()

    if not os.path.isfile(args.fn_in):
        raise FileNotFoundError(f'requested --fn-in {args.fn_in} does not exist')

    if not os.path.isdir(os.path.dirname(args.fn_out)):
        raise NotADirectoryError('--fn-out {args.fn_out} points to non-existing dir')

    if os.path.isfile(args.fn_out):
        raise FileExistsError(f'--fn-out {args.fn_out} already exists')

    return args


def merge_on_cpr_info(data: pd.DataFrame) -> pd.DataFrame:
    cpr = pd.read_stata('Y:/RegionH/SPJ/CPR info and duplicate info/spj_cprlist-plusID_180227_181106.dta')

    cpr['dob'] = cpr['cpr'].apply(lambda x: x[:-4])
    cpr['bdd'] = cpr['dob'].apply(lambda x: x[:2])

    cpr = cpr.rename(columns={'id_s': 'Id'})
    cpr = cpr.drop(columns=['id_c', 'dob', 'cpr'])

    data = data.merge(cpr, on='Id', how='left')

    return data


def load(fname: Union[str, os.PathLike], fields: Union[List[str], None]) -> pd.DataFrame:
    data = pd.read_csv(fname)
    data = data.drop_duplicates('Id')

    if 'bdd' not in data.columns: # in newer versions already merged in
        data = merge_on_cpr_info(data)

    data.loc[~data['bdd'].isnull(), 'bdd'] = data.loc[~data['bdd'].isnull(), 'bdd'].transform(lambda x: str(int(x)))

    if fields is None:
        fields = [x for x in data.columns if x.endswith('_pred')]

    data = data[['Id', 'BirthDateDay', 'bdd', *fields]]

    return data


def balance_row(
        data: pd.DataFrame,
        field: str,
        split_by: str,
        ) -> Tuple[str, float, float, int, int, float]:
    sub = data[[split_by, field]]
    sub = sub[~sub[split_by].isnull()]
    sub = sub[~sub[field].isnull()]
    sub = sub[~sub[field].isin({'0=Mangler', 'empty', 'bad cpd', 'InvalidPred'})]

    # Means, counts
    sub[field] = sub[field].astype(int)
    statistics = sub.groupby(split_by)[field].agg(['mean', 'count']).round(2)

    # P-value for different means
    test_stats = scipy.stats.ttest_ind(
        a=sub.loc[sub[split_by], field],
        b=sub.loc[sub[split_by] == False, field], # normal negate not possible as dtype is not bool pylint: disable=C0121
        )

    return [field, *statistics['mean'], *statistics['count'], round(test_stats.pvalue, 3)]


def balance_table(
        data: pd.DataFrame,
        fields: List[str],
        split_by: str,
        ) -> pd.DataFrame:
    # Maybe extract to fn with split_by var as input
    rows = []

    for field in fields:
        rows.append(balance_row(data, field, split_by))

    table = pd.DataFrame(
        rows,
        columns=['Field', 'Mean (C)', 'Mean (T)', 'Count (C)', 'Count (T)', 'P-value'],
        )

    return table


def cast_variable_binary(series: pd.Series) -> pd.Series:
    ''' If a variable -- such as lifts head (tab-b-c12-X-mo( -- is binary,
    create new series for which this holds and for which all predictions that
    # are neither 0, 1, or 2 are replaced with NaN.

    Recall that both 0 and 2 may refer to "no", depending on table type.

    '''
    mapping = {val: np.nan for val in series.unique() if val not in ('0', '1', '2', '0=Mangler')}
    mapping['0'] = '1'
    mapping['1'] = '0'
    mapping['2'] = '1'

    series = series.replace(mapping)

    return series


def get_tab_b_binary_field_names() -> List[str]:
    base_str = 'tab_b_c{}_{}_mo_pred'
    columns = [7, 9, 10, 11, 12, 13, 14]

    fields = []

    for month in (1, 2, 3, 4, 6, 9, 12):
        for col in columns:
            fields.append(base_str.format(col, month))

    return fields


def main():
    args: argparse.Namespace = parse_args()
    dst: pd.DataFrame = load(args.fn_in, args.fields)

    # [2:] excludes "Id", "BirthDateDay", and "bdd"
    fields = args.fields if args.fields else list(dst.columns)[3:]

    # Cast to uniform format (such as 3300.0 -> 3300)
    for field in ('BirthDateDay', *fields):
        not_nan = ~dst[field].isnull()
        dst.loc[not_nan , field] = dst.loc[not_nan , field].transform(rem_decimal_if_possible)

    # Grouping variable: Born 1-3
    dst['born13'] = dst['bdd'].copy()
    dst.loc[dst['bdd'].isin({'1', '2', '3'}), 'born13'] = True
    dst.loc[dst['bdd'].isin({str(i) for i in range(4, 32)}), 'born13'] = False

    # Cast the binary Table B columns to proper format
    for binary_field in get_tab_b_binary_field_names():
        dst[binary_field] = cast_variable_binary(dst[binary_field])

    # Fields to include as rows in balance table; exclude date and nurse name
    # as not directly meaningful in balance table
    balance_fields = [x for x in fields if not (x.startswith('date') or x.startswith('nn_'))]

    # Balance table by born 1-3
    table = balance_table(dst, balance_fields, 'born13')

    # Rename fields to "pretty" format
    with open('Y:/RegionH/Scripts/users/tsdj/storage/maps/map_lookup_df.pkl', 'rb') as file:
        map_lookup_df = pickle.load(file)

    table['Field'].replace(
        {
            'nn_ln_m_1_pred': 'nurse-last-name-1',
            'nn_ln_m_2_pred': 'nurse-last-name-2',
            'nn_ln_m_3_pred': 'nurse-last-name-3',
            'nn_fn_m_1_pred': 'nurse-first-name-1',
            'nn_fn_m_2_pred': 'nurse-first-name-2',
            'nn_fn_m_3_pred': 'nurse-first-name-3',
            **{v + '_pred': k for k, v in map_lookup_df.items() if not k.startswith('tab-b')},
            **{k.replace('-', '_') + '_pred': k for k, v in map_lookup_df.items() if k.startswith('tab-b')},
            },
        inplace=True
        )

    if args.fn_out.lower().endswith('.csv'):
        table.to_csv(args.fn_out, index=False)
        return

    # .tex not happy with "_" -> replace with "-"
    table['Field'] = table['Field'].str.replace('_', '-')

    with pd.option_context("max_colwidth", 1000):
        results_str = table.to_latex(
            index=False,
            escape=False,
            )

    results_str = '\n'.join(results_str.split('\n')[4:-3])

    with open(args.fn_out, 'w', encoding='utf-8') as file:
        print(results_str, file=file)


if __name__ == '__main__':
    main()
