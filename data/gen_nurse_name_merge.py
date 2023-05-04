# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Different approaches from more to gradually less stringent merge. For example:
    1) First and last name, not allowing first name to be initial. No
    duplicates with this approach, but also missing a lot of potential matches
    2) First and last name, using first name initial when only that is
    available. Technically speaking no duplicates, but in reality likely to
    match some "Emma Hansen" to "Else Hansen", for cases where only initial
    instead of full "Emma" is available and "E. Hansen" in district dataset in
    fact refers to an "Else"
    3) First name initial and last name. Duplicates now
    4) Just first name
    5) Just last name
    6) Just first name initial

"""


import os
import warnings

from typing import Dict

import pandas as pd

from create_nurse_name_lex import load_nurse_name_info_from_archive
from prepare_data_dst import prepare_nurse_firstname, prepare_nurse_lastname


def create_merge_district_to_name(info: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    value_cols = ['Kreds', 'District', 'Letter', 'Period']

    for sheet, frame in info.items():
        frame = frame.copy()

        if len(frame.columns) == 5:
            assert set(frame.columns) == set([
                'Fornavn', 'Mellemnavn', 'Efternavn', 'Gruppe', 'Year',
                ])
            # only [Fornavn, Mellemnavn, Efternavn, Gruppe, Year]
            # -> nothing in relation to group/district/etc
            continue

        frame['sheet'] = sheet
        frames.append(frame)

    districts = pd.concat(frames)

    # Joint name columns
    districts['nn'] = districts['Fornavn'] + ' ' + districts['Efternavn']
    districts['nn-s'] = districts['nn'] + '-' + districts['sheet']

    # Remove cases with bad char "?" (special char -> place \ in front)
    districts = districts[~districts['nn'].str.contains('\?')] # pylint: disable=W1401

    # Fill NAs (before duplicate drop)
    # NOTE: Did not actually check if there are any NAs, probably not
    for col in value_cols:
        # Never two conflicting values -> free to use whichever as long as not NaN
        assert (districts.groupby('nn-s')[col].nunique() < 2).all()

        # How many ffill/bffill? One of each suffices
        districts[col] = districts.groupby('nn-s')[col].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )

    # 27 cases of duplicates: Identify and drop
    # >>> (districts['nn-s'].value_counts() > 1).sum()

    # Verify when duplicate, all values are identical
    dupl = districts[districts['nn-s'].duplicated(keep=False)]
    counts = dupl.groupby('nn-s')[value_cols].nunique()
    assert (counts < 2).all().all()

    # Since always identical, fine to drop arbitrarily
    districts['is-dupl'] = districts['nn-s'].duplicated(keep='first')
    districts = districts[~districts['is-dupl']]

    # Select cols and then rehape
    districts = districts[['nn', 'Fornavn', 'Efternavn', 'Year', 'sheet', *value_cols]]

    # Pivot to wide format. By col, then merge
    wide = None

    for col in ['Year', *value_cols]:
        pivoted = districts.pivot(
            index='nn',
            columns='sheet',
            values=col,
            ).reset_index()

        pivoted.columns = ['nn'] + [f'{col}_{x}'.replace('-', '_') for x in pivoted.columns[1:]]

        if wide is None:
            wide = pivoted
        else:
            wide = wide.merge(pivoted, on='nn', how='inner')

    # Split name to first and last. Create first name initial column
    wide['nn_fn'] = wide['nn'].transform(lambda x: x.split(' ')[0])
    wide['nn_ln'] = wide['nn'].transform(lambda x: x.split(' ')[-1])

    wide = wide.drop(columns='nn')

    # Drop variables that are always NaN for one/multiple sheets
    for col in wide.columns:
        if wide[col].isnull().all():
            wide = wide.drop(columns=col)

    return wide


def load() -> pd.DataFrame:
    first_name = prepare_nurse_firstname()
    last_name = prepare_nurse_lastname()

    first_name = first_name[['Filename', *[f'nn_fn_m_{i}_pred' for i in (1, 2, 3)]]]
    last_name = last_name[['Filename', *[f'nn_ln_m_{i}_pred' for i in (1, 2, 3)]]]

    name = first_name.merge(last_name, on='Filename', how='inner')

    return name


def merge_rank_1(names: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    wide_sub = wide[wide['nn_fn'].str.len() > 1] # more than just initial

    merge = names.merge(wide_sub, on=['nn_fn', 'nn_ln'], how='left')
    merge = merge[merge['merged'] == 1.0]
    merge['rank'] = 1
    merge = merge.drop(columns='merged')

    return merge


def merge_rank_2(names: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    merge = names.merge(wide, on=['nn_fn', 'nn_ln'], how='left')
    merge = merge[merge['merged'] == 1.0]
    merge['rank'] = 2
    merge = merge.drop(columns='merged')

    return merge


def merge_rank_3(names: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    wide_initial = wide.copy()
    wide_initial = wide_initial.rename(columns={'nn_fn': 'nn_fn_i'})
    wide_initial['nn_fn_i'] = wide_initial['nn_fn_i'].transform(lambda x: x[0])

    names = names.copy()
    names['nn_fn_i'] = names['nn_fn'].transform(lambda x: x[0] if x != '0=Mangler' else x)

    merge = names.merge(wide_initial, on=['nn_fn_i', 'nn_ln'], how='left')
    merge = merge[merge['merged'] == 1.0]
    merge['rank'] = 3
    merge = merge.drop(columns=['merged', 'nn_fn_i'])

    return merge


def merge_rank_4(names: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    wide_sub = wide[wide['nn_fn'].str.len() > 1] # more than just initial

    merge = names.merge(wide_sub.drop(columns='nn_ln'), on='nn_fn', how='left')
    merge = merge[merge['merged'] == 1.0]
    merge['rank'] = 4
    merge = merge.drop(columns='merged')

    return merge


def merge_rank_5(names: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    merge = names.merge(wide.drop(columns='nn_fn'), on='nn_ln', how='left')
    merge = merge[merge['merged'] == 1.0]
    merge['rank'] = 5
    merge = merge.drop(columns='merged')

    return merge


def merge_rank_6(names: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    wide_initial = wide.copy()
    wide_initial = wide_initial.rename(columns={'nn_fn': 'nn_fn_i'})
    wide_initial['nn_fn_i'] = wide_initial['nn_fn_i'].transform(lambda x: x[0])

    names = names.copy()
    names['nn_fn_i'] = names['nn_fn'].transform(lambda x: x[0] if x != '0=Mangler' else x)

    merge = names.merge(wide_initial.drop(columns='nn_ln'), on='nn_fn_i', how='left')
    merge = merge[merge['merged'] == 1.0]
    merge['rank'] = 6
    merge = merge.drop(columns=['merged', 'nn_fn_i'])

    return merge


def create_merge(wide: pd.DataFrame, name: pd.DataFrame) -> pd.DataFrame:
    ''' Returns a dataset with number of rows per name in `name` equal to the
    total number of matches for that name across all matching methods, with
    associated "rank" for each match

    This keeps a fixed number of columns regardless of number of matches and
    leads to a dense dataset.

    '''
    map_name = {
        **{f'nn_fn_m_{number}_pred': 'nn_fn' for number in (1, 2, 3)},
        **{f'nn_ln_m_{number}_pred': 'nn_ln' for number in (1, 2, 3)},
        }
    names = pd.concat([
        name[['nn_fn_m_1_pred', 'nn_ln_m_1_pred']].rename(columns=map_name),
        name[['nn_fn_m_2_pred', 'nn_ln_m_2_pred']].rename(columns=map_name),
        name[['nn_fn_m_3_pred', 'nn_ln_m_3_pred']].rename(columns=map_name),
        ])
    names = names.drop_duplicates()
    names['nn'] = names['nn_fn'] + ' ' + names['nn_ln']
    names = names[names['nn'] != '0=Mangler 0=Mangler']

    # Now create merges
    wide['merged'] = 1.0 # Easy way to track succesful merges

    # First + last name, where first name is not 1 letter (i.e., initial)
    merge_1 = merge_rank_1(names, wide)

    # First + last name, first name can be initial
    merge_2 = merge_rank_2(names, wide)

    # First name initial and last name. Duplicates now
    merge_3 = merge_rank_3(names, wide)

    # Just first name, where first name is not 1 letter (i.e., initial)
    merge_4 = merge_rank_4(names, wide)

    # Just last name
    merge_5 = merge_rank_5(names, wide)

    # Just first name initial
    merge_6 = merge_rank_6(names, wide)

    merged = pd.concat([
        merge_1,
        merge_2,
        merge_3,
        merge_4,
        merge_5,
        merge_6,
        ])

    print(f'Total unique names in journals: {len(names)}')
    print(f'Total unique names in union of all merges: {merged["nn"].nunique()}')
    print('\nNames in journal without any merge: {}\n'.format([x for x in names['nn'] if x not in set(merged['nn'])]))

    return merged


def main():
    info = load_nurse_name_info_from_archive()
    wide = create_merge_district_to_name(info=info)

    name = load()

    # TODO add non-exact match to list? See, e.g., https://moj-analytical-services.github.io/splink/
    merged = create_merge(wide, name)
    merged = merged.drop(columns='nn') # redundant

    # TODO need to scramble names in same way as in "prepare_data_dst".
    # Potentially just do that in "prepare_data_dst.py"

    fname = os.path.join(
        r'Y:\RegionH\Scripts\users\tsdj\storage\datasets',
        'nurse-districts.csv',
        )

    if os.path.isfile(fname):
        warnings.warn('{fname} already exists, not writing')
    else:
        wide.to_csv(fname, index=False)


if __name__ == '__main__':
    main()
