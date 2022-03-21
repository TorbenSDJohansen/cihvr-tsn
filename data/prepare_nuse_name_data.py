# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:21:27 2021

@author: sa-tsdj
"""


import math
import string
import os

import pandas as pd


FN_MAIN = 'Y:/RegionH/SPJ/Database/export_181106.txt'
FN_NN_LIST = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-names-list.xlsx'
LETTERS = list(string.ascii_lowercase) + ['æ', 'ø', 'å', ' ']
LETTERS_SET = set(LETTERS)


def _search(fname: str, lname: str, names: pd.DataFrame):
    mask_fn = names['Fornavn'] == fname
    mask_ln = names['Efternavn'] == lname
    sub = names[mask_fn & mask_ln]
    
    print(sub)


def _create_lexicon():
    lex = pd.read_excel(FN_NN_LIST, sheet_name=None)
    names = []

    # TODO lowercast
    # TODO check no NaNs
    # TODO check subset LETTERS_SET (no space, etc.)
    # TODO maybe let gaard and gård be one, not sure consistent...
        # But maybe not healthy for transcription and only use post..?

    for sheet, vals in lex.items():
        print(f'\n------------\n\nSheet: {sheet}.\n')

        # Check last duplicates
        if vals['Efternavn'].value_counts().max() > 1:
            pass # there are several; print('duplicates in last name')

        vals['fn-i-ln'] = vals['Fornavn'].apply(lambda x: x[0]) + vals['Efternavn']

        if vals['fn-i-ln'].value_counts().max() > 1:
            dup = vals[vals['fn-i-ln'].duplicated(False)]
            print(f'Duplicates in initial + last name: \n\n{dup}')

        sub = vals[['Fornavn', 'Mellemnavn', 'Efternavn']].copy()
        sub['sheet'] = sheet
        names.append(sub)

    names = pd.concat(names).reset_index(drop=True)

    # TODO check across all sheets no duplicates
    # However, note that between sheet X and Y, both can contain Z (expected to)
    
    # Ideas:
        # Check all last names, if rare/obscure check proper spelling
        # Same for first names
        # SOMEHOW (not sure) check when collapse fn to fn-i, no collisions
            # -> notably collisions with OTHER sheet
    
    names['fn-i'] = names['Fornavn'].apply(lambda x: x[0])
    names['fn-i-ln'] = names['fn-i'] + ';' + names['Efternavn']
    names['fn-ln'] = names['Fornavn'] + ';' + names['Efternavn']
    
    
    
    names.groupby('fn-i-ln')['sheet'].unique()
    
    # Somehow, for each last name, check all first names and determine
    # collisions this way..?
    names.groupby('Efternavn')['Fornavn'].unique()
    names.groupby('Efternavn')['fn-i'].unique()
    
    

    # return DataFrame[fn, ln, kreds] -- more?


def _recast_name(name: str):
    if not isinstance(name, str):
        if math.isnan(name):
            return 'IsNaN;IsNaN;IsNaN'

        raise Exception(name)

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


def _get_likely_name(name_raw: str, last: bool):
    ''' Extracts last name if last is True, otherwise extracts first name.
    ALso formats the name, stripping, lower casting, etc.
    '''
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
        raise Exception(name_raw)

    k = len(names_split)

    if k == 1:
        print(f'Only one name: {name_raw}. Still accepting, using as first AND last.')
        print(name_raw)

    try:
        if last:
            candidate = names_split[-1]
        else:
            candidate = names_split[0]
    except IndexError:
        print(name_raw)

    if len(candidate) <= 2:
        print(f'Short (2 or less) length name: {name_raw}. Still accepting.')

    return candidate


def _get_likely_lastname(name_raw: str):
    return _get_likely_name(name_raw, last=True)


def _get_likely_firstname(name_raw: str):
    return _get_likely_name(name_raw, last=False)


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
        raise Exception(name_raw)

    k = len(names_split)

    if k == 1:
        print(f'Only one name: {name_raw}. Still accepting.')
        print(name_raw)

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

    # Check save/load value preservation...
    reloaded = pd.read_csv(fname)

    for col in names_merged.columns:
        if not names_merged[col].equals(reloaded[col]):
            raise Exception(col)
            # equal = names_merged[col] == reloaded[col]
            # sub1, sub2 = names_merged[~equal], reloaded[~equal]


if __name__ == '__main__':
    main()
