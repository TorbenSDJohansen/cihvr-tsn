# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import os
import pickle
import string
import warnings

from typing import Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import pandas as pd


DATASET_DIR = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets'
FN_NN_LIST = os.path.join(DATASET_DIR, 'nurse-names-list-archive.xlsx')
LABEL_DIR = r'Y:\RegionH\Scripts\data\storage\labels\keep'

LETTERS = list(string.ascii_lowercase) + ['æ', 'ø', 'å', ' ']
LETTERS_SET = set(LETTERS)

def extract_names(names: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    first_names = set()
    last_names = set()

    for name in names:
        assert set(name).issubset(LETTERS_SET)

        split = name.split(' ')

        if len(split) < 2:
            continue # not sure if first or last if only 1 name

        first = split[0]
        last = split[-1]

        last_names.add(last)

        if len(first) > 1:
            first_names.add(first) # do not add initial to list of names

    return first_names, last_names


def load_nurse_name_info_from_archive() -> Dict[str, pd.DataFrame]:
    ''' List of nurse names and other information (e.g., district), from
    archival records

    Return as dictionary of pd.DataFrames, one (key, value)-pair for each sheet
    in excel file

    '''
    info = pd.read_excel(FN_NN_LIST, sheet_name=None)
    allowed_chars = LETTERS_SET - {' '}

    # The loop below checks that:
        # Proper cols present
        # No missing first or last name
        # All chars in allowed_chars, which also implies all are lower cast
    for sheet, vals in info.items():
        assert {'Fornavn', 'Mellemnavn', 'Efternavn', 'Year'}.issubset(vals.columns)
        assert 'Gruppe' in vals.columns or 'Kreds' in vals.columns or 'Period' in vals.columns or 'District' in vals.columns
        assert len({'District', 'Letter'} & set(vals.columns)) != 1 # none or both present

        if len({'District', 'Letter'} & set(vals.columns)) == 2:
            assert len(vals.columns) <= 7
        else:
            assert len(vals.columns) <= 5

        assert vals['Fornavn'].isnull().sum() + vals['Efternavn'].isnull().sum() == 0

        for col in ['Fornavn', 'Mellemnavn', 'Efternavn']:
            sub = vals.loc[~vals[col].isnull(), col]

            if sheet == '0172-0173' and col == 'Fornavn':
                sub = sub[sub != '?']

            assert sub.apply(lambda x: set(x).issubset(allowed_chars)).all()

    return info


def load_nurse_name_list_from_labels() -> np.ndarray[str]:
    labels = []

    for split in ('train', 'test'):
        for i in (1, 2, 3):
            fname = os.path.join(LABEL_DIR, split, f'nurse-name-{i}.npy')
            labels.append(np.load(fname, allow_pickle=True))

    labels = np.concatenate(labels)

    labels = pd.DataFrame(labels, columns=['fname', 'label'])
    labels = labels[~labels['label'].isin(['bad cpd', '0=Mangler'])]

    return labels['label'].unique()


def save_lex(fname: Union[str, os.PathLike], lex: Set[str]):
    if not os.path.isfile(fname):
        with open(fname, 'wb') as file:
            pickle.dump(lex, file)
    else:
        warnings.warn(f'File "{fname}" already exists! Not writing.')


def create_lex(
        nurse_names: List[Iterable[str]],
        out_dir: Union[str, os.PathLike],
        ):
    first_names = set()
    last_names = set()

    for names in nurse_names:
        first_names_, last_names_ = extract_names(names)

        first_names = first_names.union(first_names_)
        last_names = last_names.union(last_names_)

    save_lex(os.path.join(out_dir, 'fn.pkl'), first_names)
    save_lex(os.path.join(out_dir, 'ln.pkl'), last_names)


def nurse_info_to_nurse_names(nurse_info: Dict[str, pd.DataFrame]) -> np.ndarray[str]:
    names = pd.concat( # extract as fn prob...
        [df[['Fornavn', 'Efternavn']] for df in nurse_info.values()]
        )
    names['nurse-name'] = names['Fornavn'] + ' ' + names['Efternavn']
    names = names[~names['nurse-name'].isin(['bad cpd', '0=Mangler'])]

    # Remove cases with bad char "?" (special char -> place \ in front)
    names = names[~names['nurse-name'].str.contains('\?')] # pylint: disable=W1401

    return names['nurse-name'].unique()


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
            # -> nothing in relation to group
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

    wide['nn_fn_i'] = wide['nn_fn'].transform(lambda x: x[0])

    # Check duplicated with fn initial + ln-> multiple cases
    # >>> wide[['nn_fn_i', 'nn_ln']].duplicated().sum()

    wide = wide.drop(columns='nn')

    # Drop variables that are always NaN for one/multiple sheets
    for col in wide.columns:
        if wide[col].isnull().all():
            wide = wide.drop(columns=col)

    fname = os.path.join(DATASET_DIR, 'nurse-districts.csv')

    if os.path.isfile(fname):
        warnings.warn('{fname} already exists, not writing')
    else:
        wide.to_csv(fname, index=False)


def main():
    nurse_info = load_nurse_name_info_from_archive()
    nurse_label_names = load_nurse_name_list_from_labels()

    # Iterable of nurse names form `nurse_info`
    nurse_info_names = nurse_info_to_nurse_names(nurse_info)

    # Create lexicon
    create_lex(
        nurse_names=[nurse_info_names, nurse_label_names],
        out_dir=r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex',
        )

    # Create merge district to name set
    create_merge_district_to_name(info=nurse_info)


if __name__ == '__main__':
    main()
