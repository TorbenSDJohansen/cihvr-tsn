# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import math
import string
import os
import pickle

import pandas as pd


FN_MAIN = 'Y:/RegionH/SPJ/Database/export_181106.txt'
FN_NN_LIST = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-names-list.xlsx'
LETTERS = list(string.ascii_lowercase) + ['æ', 'ø', 'å', ' ']
LETTERS_SET = set(LETTERS)


def _load():
    lex = pd.read_excel(FN_NN_LIST, sheet_name=None)
    allowed_chars = LETTERS_SET - {' '}

    # The loop below checks that:
        # Proper cols present
        # No missing first or last name
        # All chars in allowed_chars, which also implies all are lower cast
    for sheet, vals in lex.items():
        assert {'Fornavn', 'Mellemnavn', 'Efternavn', 'Year'}.issubset(vals.columns)
        assert 'Gruppe' in vals.columns or 'Kreds' in vals.columns or 'Period' in vals.columns or len(vals.columns) == 4
        assert len(vals.columns) <= 5

        assert vals['Fornavn'].isnull().sum() + vals['Efternavn'].isnull().sum() == 0

        for col in ['Fornavn', 'Mellemnavn', 'Efternavn']:
            sub = vals.loc[~vals[col].isnull(), col]

            if sheet == '0172-0173' and col == 'Fornavn':
                sub = sub[sub != '?']

            assert sub.apply(lambda x: set(x).issubset(allowed_chars)).all()

    return lex


def _create_lexicon():
    # Interested in three things, first two easiest:
        # 1) All unique last names
        # 2) All unique first names
        # 3) All unique combinations

    lex_dfs = _load()
    names = []

    for sheet, vals in lex_dfs.items():
        if sheet == '0172-0173':
            continue # Can only use as additional names, contains duplicates

        # Check last name duplicates
        if vals['Efternavn'].value_counts().max() > 1:
            pass # there are several; print('duplicates in last name')

        vals['fn-ln'] = vals['Fornavn'] + vals['Efternavn']
        vals['fn-i-ln'] = vals['Fornavn'].apply(lambda x: x[0]) + vals['Efternavn']

        # Assert no duplicates in first + last name
        assert vals['fn-ln'].value_counts().max() == 1, sheet

        if vals['fn-i-ln'].value_counts().max() > 1:
            dup = vals[vals['fn-i-ln'].duplicated(False)]
            print(f'\nDuplicates for {sheet} in initial + last name: \n\n{dup}')

        sub = vals[['Fornavn', 'Mellemnavn', 'Efternavn']].copy()
        sub['sheet'] = sheet
        names.append(sub)

    names = pd.concat(names).reset_index(drop=True)

    # Ideas:
        # SOMEHOW (not sure) check when collapse fn to fn-i, no collisions
            # -> notably collisions with OTHER sheet

    # Useful to manually check last names (if one case only, more likely wrong)
    ln_count = names['Efternavn'].value_counts().reset_index().rename(
        columns={'index': 'Efternavn', 'Efternavn': 'ln_count'},
        )
    names = names.merge(ln_count, on='Efternavn', how='left')

    # Useful to manually check first names (if one case only, more likely wrong)
    fn_count = names['Fornavn'].value_counts().reset_index().rename(
        columns={'index': 'Fornavn', 'Fornavn': 'fn_count'},
        )
    names = names.merge(fn_count, on='Fornavn', how='left')

    # Potentially other useable last names in lex_dfs['0172-0173']['Efternavn']),
    # but turns out not to be the case
    assert set(lex_dfs['0172-0173']['Efternavn']) - set(names['Efternavn']) == set()

    lex_last = set(names['Efternavn'])
    lex_first = set([x for x in names['Fornavn'] if len(x) > 1])
    lex_first_i = set(names['Fornavn'].apply(lambda x: x[0])) # Perhaps include all/more letters?

    return {
        'ln': lex_last,
        'fn': lex_first,
        'fn-i': lex_first_i,
        }


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

    return ' '.join((names_split))


def _get_unique_names(names: pd.DataFrame, col: str, last: bool):
    sub = names[col]
    sub = sub[~sub.isnull()]
    sub = sub[sub != '0=Mangler']
    split = sub.str.split(' ')

    if last:
        unique_names = split.apply(lambda x: x[-1]).unique()
    else:
        split = split[split.apply(lambda x: len(x) > 1)] # drop when only one name
        unique_names = split.apply(lambda x: x[0]).unique()

    return unique_names


def _get_all_unique_names(names, last: bool):
    unique_names = set()

    for i in (1, 2, 3):
        _unique = set(_get_unique_names(names, f'nurse-name-{i}', last))
        unique_names = unique_names.union(_unique)

    return unique_names


def _save_lex(lex: set, path: str, fname: str):
    fname_full = os.path.join(path, fname)

    if not os.path.isfile(fname_full):
        with open(fname_full, 'wb') as file:
            pickle.dump(lex, file)
    else:
        print(f'WARNING: File "{fname_full}" already exists! Not writing.')


def _save_lexicons(lex: dict, last_names: set, first_names: set):
    assert set(lex.keys()) == {'ln', 'fn', 'fn-i'}
    path = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex'

    # Different levels of strictness:
        # 1) Most strict: Only use lex
        # 2) Least strict: Include also ALL names in last_names
        # 3) Some combinations, probably dropping rare names from other...

    lex_ln_loose = lex['ln'].union(last_names)

    # All all single letters, initials often used.
    lex_fn_strict = lex['fn'].union(LETTERS_SET - {' '})
    lex_fn_loose = lex_fn_strict.union(first_names)

    _save_lex(lex['ln'], path, 'ln-strict.pkl')
    _save_lex(lex_ln_loose, path, 'ln-loose.pkl')

    _save_lex(lex_fn_strict, path, 'fn-strict.pkl')
    _save_lex(lex_fn_loose, path, 'fn-loose.pkl')



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

    lex = _create_lexicon()
    unique_last_names = _get_all_unique_names(names_merged, last=True)
    unique_first_names = _get_all_unique_names(names_merged, last=False)

    _save_lexicons(lex, last_names=unique_last_names, first_names=unique_first_names)

    # TODO if a name is rare in labels and a close match to either something in
    # `lex` or another, non-rare, label, it is likely incorrect. Maybe use this
    # information to, e.g., match to nearest valid

    fname = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names.csv'

    if not os.path.isfile(fname):
        names_merged.to_csv(fname, index=False)
    else:
        print(f'WARNING: File "{fname}" already exist - not writing new file!')

    # Check save/load value preservation...
    reloaded = pd.read_csv(fname)

    for col in names_merged.columns:
        if not names_merged[col].equals(reloaded[col]):
            raise Exception(col)
            # equal = names_merged[col] == reloaded[col]
            # sub1, sub2 = names_merged[~equal], reloaded[~equal]


if __name__ == '__main__':
    main()
