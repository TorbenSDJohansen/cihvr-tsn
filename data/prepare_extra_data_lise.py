# -*- coding: utf-8 -*-
r"""
Prepares new data at Y:\RegionH\SPJ\EkstraDataFraLise

@author: sa-tsdj
"""


import os
import warnings

from typing import Set, Dict

import numpy as np
import pandas as pd

from prepare_nurse_name_data import _format_name
from gen_map_lookup_df import gen_map_name


FN_MAIN = 'Y:/RegionH/SPJ/Database/export_181106.txt' # to merge on
PATH = r'Y:\RegionH\SPJ\EkstraDataFraLise'
FN_CC = os.path.join(PATH, 'CIHVR_forCCsample_280322_final.dta')
FN_CPC = os.path.join(PATH, 'CIHVR_forCPC_250322_final.dta')

def _merge_filename_on(df: pd.DataFrame, df_main: pd.DataFrame):
    # Verify unique IDs
    assert len(df) == df['id'].nunique() + df['id'].isna().sum()

    # Retype id to str without ".0" part
    df['id'] = df['id'].apply(lambda x: x if np.isnan(x) else str(int(x)))
    # NOTE: We do not drop leading 0s, as 'id' is 1, 2, 3, ...., n

    # Prepare filename set
    sub = df_main[['Filename', 'Id']]
    sub = sub.rename(columns={'Id': 'id'})
    sub['id'] = sub['id'].astype(str)

    merged = df.merge(sub, on='id', how='inner')

    # For good measure, assert all unique
    assert len(merged) == merged['Filename'].nunique()

    # For CPC and CC, respectively
    assert len(merged) == 4081 or len(merged) == 2372

    return merged


def _recast_name(name: str):
    # Mostly similar to `from prepare_nuse_name_data import _recast_name`
    if not isinstance(name, str):
        raise TypeError(f'name {name} is not str')

    if name == '':
        return 'IsNaN;IsNaN;IsNaN'

    # # Special case typo fixing...
    if name == 'P. Lund; M. Ravnborg, M. L. Poort':
        name = name.replace(',', ';')

    # Special case trailing ';'
    if name[-1] == ';':
        name = name[:-1]

    name_split = name.split(';')

    if len(name_split) > 3:
        print(name)

    empty = ['0=Mangler'] * (3 - len(name_split))

    name_mod = ';'.join(name_split + empty)

    return name_mod


def prepare_nurse_names(df_cpc, df_cc, overlap):
    # Inspection of nurse names - when overlap, always identical between two!
    sub_cc = df_cc.loc[df_cc['Filename'].isin(overlap), ['Filename', 'sundh_plj']]
    sub_cpc = df_cpc.loc[df_cpc['Filename'].isin(overlap), ['Filename', 'sundh_plj']]
    sub_m = sub_cc.merge(sub_cpc, on='Filename', how='inner')
    sub_m['name_is_eq'] = sub_m['sundh_plj_x'] == sub_m['sundh_plj_y']
    assert sub_m['name_is_eq'].all()

    # Create set of nurse names by concat -> drop not-unique (doable by above)
    new_nurse_names = pd.concat([
        df_cpc[['Filename', 'sundh_plj']],
        df_cc[['Filename', 'sundh_plj']],
        ]).copy()
    new_nurse_names = new_nurse_names.drop_duplicates('Filename')

    # Recast new nurse names to match format
    new_nurse_names['name'] = list(map(_recast_name, new_nurse_names['sundh_plj']))

    # 3xNaN below are NOT 3x empty, we do not know what is there -> drop
    new_nurse_names = new_nurse_names[new_nurse_names['name'] != 'IsNaN;IsNaN;IsNaN']

    # Few with more than 4 seperate nurses. Still can use first three, tested
    # by manual inspection.
    # >> new_nurse_names['4 or more names'] = new_nurse_names['name'].str.split(';').apply(lambda x: len(x) >= 4)
    # Might be a line with two names in these cases

    for name_number in range(1, 4):
        new_nurse_names[f'name{name_number}'] = new_nurse_names['name'].str.split(';').apply(
            lambda x: x[name_number - 1] # pylint: disable=W0640
            )
        new_nurse_names[f'nurse-name-{name_number}'] = list(map(
            _format_name, new_nurse_names[f'name{name_number}'].values,
            ))

    # Check which of these filenames NOT in already obtained nurse names. Large
    # overlap but also some new files.
    old_nn = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names.csv')

    # Overlap with those we already have
    overlap_to_old = set(new_nurse_names['Filename']).intersection(old_nn['Filename'])
    assert len(overlap_to_old) == 4777

    # When overlap, same name or not? Yes, always. New names useless.
    m_overlap = old_nn[['Filename', 'name']].merge(
        new_nurse_names[['Filename', 'name']], on='Filename', how='inner'
        )
    m_overlap['name_is_eq'] = m_overlap['name_x'] == m_overlap['name_y']
    assert sub_m['name_is_eq'].all()

    # Where we do NOT have overlap. Only 47 -- really not much new here.
    new_unique_filenames = set(new_nurse_names['Filename']) - set(old_nn['Filename'])
    assert len(new_unique_filenames) == 47

    # Save the NEW ones
    fname = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names_additional_new_lise_data.csv'
    new_nurse_names_unique = new_nurse_names[new_nurse_names['Filename'].isin(new_unique_filenames)]
    new_nurse_names_unique = new_nurse_names_unique.reset_index(drop=True)

    if not os.path.isfile(fname):
        new_nurse_names_unique.to_csv(fname, index=False)
    else:
        warnings.warn(f'File "{fname}" already exist - not writing new file!')

    # Check save/load value preservation...
    reloaded = pd.read_csv(fname)

    for col in new_nurse_names_unique.columns:
        if not new_nurse_names_unique[col].equals(reloaded[col]):
            raise ValueError(col)


def rename_cols(frame: pd.DataFrame, map_name: Dict[str, str]) -> pd.DataFrame:
    assert set(map_name.keys()).issubset(frame.columns), set(map_name.keys()) - set(frame.columns)

    return frame.rename(columns=map_name)


def inspect_tab_b_cell_overlap(
        df_cpc: pd.DataFrame,
        df_cc: pd.DataFrame,
        overlap: Set[str],
        name: str,
        ):
    ''' Inspect for equality between both datasets in their overlap where both
    are non-missing
    '''
    sub_cc = df_cc.loc[df_cc['Filename'].isin(overlap), ['Filename', name]]
    sub_cpc = df_cpc.loc[df_cpc['Filename'].isin(overlap), ['Filename', name]]
    sub_m = sub_cc.merge(sub_cpc, on='Filename', how='inner').dropna()
    sub_m['name_is_eq'] = sub_m[name + '_x'] == sub_m[name + '_y']

    assert sub_m['name_is_eq'].all(), sub_m[~sub_m['name_is_eq']]


def prepare_tab_b(
        df_cpc: pd.DataFrame,
        df_cc: pd.DataFrame,
        overlap: Set[str],
        map_name: Dict[str, str],
        ):
    # Recast some inspected values which were labelled wrongly
    df_cc.loc[df_cc['Filename'] == 'SPJ_2014-07-08_0210.PDF', 'carev1'] = 1
    df_cc.loc[df_cc['Filename'] == 'SP2_36639.pdf', 'smiles_v4'] = 1
    df_cc.loc[df_cc['Filename'] == 'SPJ_2014-07-08_0210.PDF', 'babbles_v3'] = 2
    df_cc.loc[df_cc['Filename'] == 'SP2_36639.pdf', 'babbles_v4'] = 1
    df_cc.loc[df_cc['Filename'] == 'SPJ_2014-07-08_0210.PDF', 'bvf3'] = 1
    df_cc.loc[df_cc['Filename'] == 'SPJ_2014-07-08_0210.PDF', 'bvf4'] = 1
    df_cc.loc[df_cc['Filename'] == 'SPJ_2014-07-08_0210.PDF', 'nb_meals_v2'] = 5 # image says 5-6, this (CC) says 6, CPC 5, change this to 5

    # Inspection of overlap - when overlap and both non-missing, always equal
    for name in map_name.values():
        inspect_tab_b_cell_overlap(df_cpc, df_cc, overlap, name)

    # For `df_cpc`, if column "no_weight_data" equals 1, we know all Table B
    # cells are empty
    df_cpc.loc[df_cpc['no_weight_data'] == 1, list(map_name.values())] = '0=Mangler'

    # Concatenate - note we now have 117 duplicates for 'Filename'
    tab_b = pd.concat([
        df_cpc[['Filename', *map_name.values()]],
        df_cc[['Filename', *map_name.values()]],
        ]).copy()

    # We want to drop the *right* duplicates, i.e. those with missing if the
    # other df then is non-missing at that pos. To see the problem
    # tab_b.loc[tab_b['Filename'] == 'SP2_37259.pdf', 'harmonyv2']
    # >>>
    # 198     3.0
    # 2027    NaN
    # Name: harmonyv2, dtype: float64
    # Here we want to replace the NA with 3.0

    grouped = tab_b[tab_b['Filename'].isin(overlap)].groupby('Filename')
    filled = grouped.apply(lambda x: x.fillna(x.mean()))
    filled = filled.drop_duplicates('Filename')

    # Drop duplicate cases, then append the filled version
    tab_b = tab_b[~tab_b['Filename'].isin(overlap)]
    tab_b = pd.concat([tab_b, filled])
    tab_b = tab_b.reset_index(drop=True)

    fname = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\tab_b_additional_new_lise_data.csv'

    if not os.path.isfile(fname):
        tab_b.to_csv(fname, index=False)
    else:
        warnings.warn(f'File "{fname}" already exist - not writing new file!')

    # Check save/load value preservation...
    reloaded = pd.read_csv(fname)

    for col in tab_b.columns:
        if not tab_b[col].equals(reloaded[col]):
            raise ValueError(col)

    # Lise recommends maybe not use any of v3 (1 mo), highest prob wrong.
    # Handle this in `gen_labels.py`: In cases where overlap with old sample,
    # drop the new part. In case no overlap then simply never write a label
    # file at all


def main():
    df_main = pd.read_csv(FN_MAIN, sep=';', na_values=[''], keep_default_na=False)
    df_cc = pd.read_stata(FN_CC)
    df_cpc = pd.read_stata(FN_CPC)

    df_cc = _merge_filename_on(df_cc, df_main)
    df_cpc = _merge_filename_on(df_cpc, df_main)

    map_name = gen_map_name()
    df_cc = rename_cols(df_cc, map_name)
    df_cpc = rename_cols(df_cpc, map_name)

    # Check overlap between the two - do they refer to same files? 117 do
    overlap = set(df_cc['Filename']).intersection(df_cpc['Filename'])
    assert len(overlap) == 117

    # Prepare new nurse name data (really not much, see fn)
    prepare_nurse_names(df_cpc, df_cc, overlap)

    # Prepare new Table B data
    prepare_tab_b(df_cpc, df_cc, overlap, map_name)

    # Also take weights, dates? Perhaps not worth as quite noisy data and for
    # those columns we already have enough training data to train models

    # At some point, need to decide whether to use variable from this set when
    # also present in main. Probably do NOT do this, only add to main when not
    # already present in main.


if __name__ == '__main__':
    main()
