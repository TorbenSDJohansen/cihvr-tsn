# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import math
import string
import os
import warnings

import pandas as pd


FN_MAIN = 'Y:/RegionH/SPJ/Database/export_181106.txt'
FN_NN_LIST = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-names-list.xlsx'
LETTERS = list(string.ascii_lowercase) + ['æ', 'ø', 'å', ' ']
LETTERS_SET = set(LETTERS)


def _recast_name(name: str):
    if not isinstance(name, str):
        if math.isnan(name):
            return 'IsNaN;IsNaN;IsNaN'

        raise TypeError(f'name {name} it not str')

    # Special case typo fixing...
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


def _merge_filename_on(names: pd.DataFrame):
    main_df = pd.read_csv(FN_MAIN, sep=';')
    main_df = main_df[['Filename', 'JournalId']]
    main_df = main_df.rename(columns={'JournalId': 'jnr'})
    main_df['jnr'] = main_df['jnr'].astype(str)
    main_df['jnr'] = main_df['jnr'].apply(lambda x: '-'.join((x[:2], x[2:])))

    names_merged = names.merge(main_df, on='jnr', how='left')

    return names_merged


def _format_name(name_raw: str):
    if name_raw.strip() == '0=Mangler':
        return '0=Mangler'

    name = name_raw.replace('.', ' ')
    name = name.replace('-', '')
    name = name.replace('ü', 'u')
    name = name.strip()
    name = name.lower()
    names_split = name.split()

    if '?' in name:
        print(f'Unreadable name: {name_raw}. Returning None.')
        return None

    if '(vikar)' in name:
        print(f'Vikar: {name_raw}. Returning None.')
        return None

    if not set(name).issubset(LETTERS_SET):
        raise ValueError(f'name_raw {name_raw} -> {name} contains chars not part of {LETTERS_SET}')

    k = len(names_split)

    if k == 1:
        print(f'Only one name: {name_raw}. Still accepting.')

    return ' '.join((names_split))


def main():
    """
    Creates a cleaned dataset of nurse first and last names.

    Raises
    ------
    Exception
        Raises exception if save/load does not preserve values.

    Returns
    -------
    None.

    """
    names = pd.read_excel(
        r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\SPJnames_occupation_all2.xls',
        )
    names = names[['jnr', 'sundh_plj']]
    names['name'] = list(map(_recast_name, names['sundh_plj']))

    # 3xNaN below are NOT 3x empty, we do not know what is there -> drop
    names = names[names['name'] != 'IsNaN;IsNaN;IsNaN']

    # Some duplicate journal numbers! From manual inspection, appears entries
    # accidently entered multiple times. Also evident when performing
    # drop_duplicates on all rows. See line below to check
    # names['v'] = names.groupby('jnr')['jnr'].transform('count')
    names = names.drop_duplicates()

    # Few with more than 4 seperate nurses. Still can use first three, tested
    # by manual inspection.
    # names['4 or more names'] = names['name'].str.split(';').apply(lambda x: len(x) >= 4)

    for name_number in range(1, 4):
        names[f'name{name_number}'] = names['name'].str.split(';').apply(
            lambda x: x[name_number - 1] # pylint: disable=W0640
            )
        names[f'nurse-name-{name_number}'] = list(map(
            _format_name, names[f'name{name_number}'].values,
            ))

    names_merged = _merge_filename_on(names)

    # Some not merged, i.e. we do not have Filename. These are unuseable.
    names_merged = names_merged[~names_merged['Filename'].isna()]

    # Some duplicate journal numbers, as multiple Filename for each jnr.
    # Currently keeping those still, in either case not too many. See below
    # names_merged['v'] = names_merged.groupby('jnr')['jnr'].transform('count')

    names_merged = names_merged.reset_index(drop=True)

    fname = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names.csv'

    if not os.path.isfile(fname):
        names_merged.to_csv(fname, index=False)
    else:
        warnings.warn(f'File "{fname}" already exist - not writing new file!')

    # Check save/load value preservation...
    reloaded = pd.read_csv(fname)

    for col in names_merged.columns:
        if not names_merged[col].equals(reloaded[col]):
            raise ValueError(col)
            # equal = names_merged[col] == reloaded[col]
            # sub1, sub2 = names_merged[~equal], reloaded[~equal]


if __name__ == '__main__':
    main()
