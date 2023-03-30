# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Script to compare predictions for DST upload. Useful to provide potential issue
with an unexpected large deviation in transcriptions between two version of
data upload to DST.

While some deviation is expected, especially in terms of coverage but also in
terms of improved performance meaning that we expect revised transcriptions to
be different from an older upload, for most columns we do not expect this
difference to be very large.

"""


import argparse
import os

from typing import List, Union

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--fn-old', type=str)
    parser.add_argument('--fn-new', type=str)
    parser.add_argument('--fields', type=str, nargs='+', default=None)
    parser.add_argument('--fn-out', type=str)

    args = parser.parse_args()

    if not os.path.isfile(args.fn_old):
        raise FileNotFoundError(f'requested --fn-old {args.fn_old} does not exist')

    if not os.path.isfile(args.fn_new):
        raise FileNotFoundError(f'requested --fn-new {args.fn_new} does not exist')

    if not os.path.isdir(os.path.dirname(args.fn_out)):
        raise NotADirectoryError('--fn-out {args.fn_out} points to non-existing dir')

    if os.path.isfile(args.fn_out):
        raise FileExistsError(f'--fn-out {args.fn_out} already exists')

    return args


def load(fname: Union[str, os.PathLike], fields: Union[List[str], None]) -> pd.DataFrame:
    data = pd.read_csv(fname)
    data = data.drop_duplicates('Id')

    if fields is None:
        fields = [x for x in data.columns if x.endswith('_pred')]

    data = data[['Id', *fields]]

    return data


def rem_decimal_if_possible(value: Union[str, int, float]) -> str:
    if isinstance(value, float):
        if int(value) == value: # 0.0 -> "0"
            return str(int(value))

        raise ValueError(value)

    if isinstance(value, int): # 0 -> "0"
        return str(value)

    try:
        return rem_decimal_if_possible(float(value))
    except ValueError:
        return value


def compare_field(
        old: pd.DataFrame,
        new: pd.DataFrame,
        field: str,
        ):
    if field not in old.columns:
        return [field, 0, len(new) - new[field].isnull().sum(), None]

    if field not in new.columns:
        return [field, len(old) - old[field].isnull().sum(), 0, None]

    merged = old[['Id', field]].merge(new[['Id', field]], on='Id', how='inner')

    assert len(merged) == len(old) == len(new)

    # Compare coverage
    cov_old = len(old) - old[field].isnull().sum()
    cov_new = len(new) - new[field].isnull().sum()

    # Look at (dis)similarities when both non-missing
    overlap = merged[~merged.isnull().any(axis=1)].copy()

    # Make sure values such as 3150 and 3150.0 are treated as equal
    overlap[field + '_x'] = overlap[field + '_x'].transform(rem_decimal_if_possible)
    overlap[field + '_y'] = overlap[field + '_y'].transform(rem_decimal_if_possible)

    overlap['identical'] = overlap[field + '_x'] == overlap[field + '_y']
    share_identical = overlap['identical'].mean()

    return [field, cov_old, cov_new, share_identical]


def main():
    args: argparse.Namespace = parse_args()

    old: pd.DataFrame = load(args.fn_old, args.fields)
    new: pd.DataFrame = load(args.fn_new, args.fields)

    fields = args.fields if args.fields else list(new.columns)[1:] # [1:] drops "Id"
    results = []

    for field in fields:
        results.append(compare_field(old, new, field))

    results = pd.DataFrame(results, columns=['field', 'Cov (old)', 'Cov (new)', 'Share similar'])
    results['Change cov.'] = results['Cov (new)'] / results['Cov (old)'] - 1

    results['Share similar'] = (100 * results['Share similar']).round(1)
    results['Change cov.'] = (100 * results['Change cov.']).round(1)

    results.to_csv(args.fn_out, index=False)


if __name__ == '__main__':
    main()
