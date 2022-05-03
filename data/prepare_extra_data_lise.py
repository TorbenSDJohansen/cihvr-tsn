# -*- coding: utf-8 -*-
r"""
Prepares new data at Y:\RegionH\SPJ\EkstraDataFraLise

@author: sa-tsdj
"""


import os

import numpy as np
import pandas as pd

from prepare_nuse_name_data import _format_name


FN_MAIN = 'Y:/RegionH/SPJ/Database/export_181106.txt' # to merge on
PATH = r'Y:\RegionH\SPJ\EkstraDataFraLise'
FN_CC = os.path.join(PATH, 'CIHVR_forCCsample_280322_final.dta')
FN_CPC = os.path.join(PATH, 'CIHVR_forCPC_250322_final.dta')


MAP_VARNAME = {
     # TODO varname from df_cc/df_cpc to format of df_main - or LIKE that, not
     # all vars in new sets present in old!!
    }

MAP_VARVAL = {
    # TODO like 3.0 might be 3=Kunstig ernæring, etc.
    }

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
    # Mostly similar to from prepare_nuse_name_data import _recast_name
    if not isinstance(name, str):
        raise Exception(name)

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
        print(f'WARNING: File "{fname}" already exist - not writing new file!')

    # Check save/load value preservation...
    reloaded = pd.read_csv(fname)

    for col in new_nurse_names_unique.columns:
        if not new_nurse_names_unique[col].equals(reloaded[col]):
            raise Exception(col)


def main():
    df_main = pd.read_csv(FN_MAIN, sep=';', na_values=[''], keep_default_na=False)
    df_cc = pd.read_stata(FN_CC)
    df_cpc = pd.read_stata(FN_CPC)

    df_cc = _merge_filename_on(df_cc, df_main)
    df_cpc = _merge_filename_on(df_cpc, df_main)

    # Check overlap between the two - do they refer to same files? 117 do
    overlap = set(df_cc['Filename']).intersection(df_cpc['Filename'])
    assert len(overlap) == 117

    # Prepare new nurse name data (really not much, see fn)
    prepare_nurse_names(df_cpc, df_cc, overlap)

    # When overlap, need decision rule - where to draw variable from?

    # At some point, need to decide whether to use variable from this set when
    # also present in main. Probably do NOT do this, only add to main when not
    # already present in main.


if __name__ == '__main__':
    main()
