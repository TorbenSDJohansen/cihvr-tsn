# -*- coding: utf-8 -*-
"""
@author: tsdj

Heavily updated to reflect this new segmentation. This has large consequences
for "bad cpd" (or more broadly "bad segmentation"); current labels of these are
no longer useful.

This likely will lead to a need for an iterative change to catch "bad
segmentation" examples and add those.

"""



import os
import pickle
import warnings

from typing import Callable, Dict, Set

import json

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from verifiers import Verifiers


def load_mod_weight() -> pd.DataFrame:
    ''' With new segmentation, the old "bad cpd" values are no longer
    meaningful; they might very well be properly segmented now. For that
    reason, drop fields from all journals marked as "bad cpd".

    Note that empty values are still very useful to add in this way.

    '''
    map_raw_to_final = {'0': 'empty', 0: 'empty', '1': 'bad cpd', 1: 'bad cpd'}

    with open('Y:/RegionH/Scripts/users/cmd/BW_low_confidence5200.json', 'rb') as file:
        entires_new = json.load(file)

    entries = entires_new['elements'][:entires_new['cursor']]
    new_label_info = {'fname': [], 'cell': [], 'label': []}

    for entry in entries:
        label_raw = entry['properties']['Birth Weight']

        if not label_raw in map_raw_to_final:
            print(entry)
            continue

        new_label_info['fname'].append(entry['name'])
        new_label_info['cell'].append(entry['folder'].split('\\')[-1])
        new_label_info['label'].append(map_raw_to_final[label_raw])

    new_label_info = pd.DataFrame(new_label_info)

    # Now drop all cells from all journals where at least one field labelled as
    # "bad cpd". This is needed as new segmentation means this is no longer
    # meaningful.
    bad_cpd_files = set(new_label_info[new_label_info['label'] == 'bad cpd']['fname'])
    new_label_info = new_label_info[~new_label_info['fname'].isin(bad_cpd_files)]

    # Now only "empty" cases exist (allows us to pass assert below)
    assert (new_label_info['label'] == 'empty').all()

    # Obtain column Journal (journal filename) -> drop by-page column fname
    new_label_info['Journal'] = new_label_info['fname'].transform(lambda x: x.split('.')[0])
    new_label_info = new_label_info.drop(columns='fname')

    return new_label_info


def load_100x112_tab_b_sample_journals() -> Set[str]:
    folder = r'Y:\RegionH\Scripts\data\storage\labels\tab-b-100x112-test-set\test'
    files = [os.path.join(folder, x) for x in os.listdir(folder)]
    labels_test_100x112 = np.concatenate([np.load(x, allow_pickle=True) for x in files])
    files_test_100x112 = [x.split('.jpg')[0] for x in labels_test_100x112[:, 0]]
    files_test_100x112 = set(files_test_100x112)

    assert len(files_test_100x112) == 100

    return files_test_100x112


def merge_nurse_names(
        df_main: pd.DataFrame,
        fn_df_nurse_names: str,
        fn_df_nurse_names_new: str,
        ) -> pd.DataFrame:
    nurse_names = pd.read_csv(fn_df_nurse_names)
    nurse_names_new = pd.read_csv(fn_df_nurse_names_new)

    # TODO prob add @malthe set here, then add to intersect check and to concat

    assert set(nurse_names['Filename']).intersection(nurse_names_new['Filename']) == set()

    nurse_names = pd.concat([
        nurse_names[['Filename', 'nurse-name-1', 'nurse-name-2', 'nurse-name-3']],
        nurse_names_new[['Filename', 'nurse-name-1', 'nurse-name-2', 'nurse-name-3']],
        ])

    # As identifier use journal name without file suffix
    nurse_names['Journal'] = nurse_names['Filename'].transform(lambda x: x.split('.')[0])
    nurse_names = nurse_names.drop(columns='Filename')

    df_main = df_main.merge(nurse_names, on='Journal', how='left')

    return df_main


def merge_new_table_b(
        df_main: pd.DataFrame,
        fn_df_tab_b_new: str,
        map_lookup_df: Dict[str, str],
        ) -> pd.DataFrame:
    tab_b = pd.read_csv(fn_df_tab_b_new)

    # As identifier use journal name without file suffix
    tab_b['Journal'] = tab_b['Filename'].transform(lambda x: x.split('.')[0])
    tab_b = tab_b.drop(columns='Filename')

    # Drop all the 1-month rows (columns in DataFrame) as suggested by Lise
    bad_rows = [map_lookup_df[f'tab-b-c{x}-1-mo'] for x in range(1, 17)]
    tab_b = tab_b.drop(columns=bad_rows)

    # Drop all rows corresponding to any journal part of manual 100x112 sample
    journals_in_tab_b_100x112_test = load_100x112_tab_b_sample_journals()
    tab_b = tab_b[~tab_b['Journal'].isin(journals_in_tab_b_100x112_test)]

    # Identify which columns are not already in df_main
    tab_b_new_cols = ['Journal'] + [x for x in tab_b.columns if x not in df_main.columns]

    # Safe with normal merge for cols not already in df_main
    df_main = df_main.merge(tab_b[tab_b_new_cols], on='Journal', how='left')

    # For those already in df_main, only replace when no value. Useful to let
    # `tab_b` index be filename to match filenames between the two data frames
    tab_b.index = tab_b['Journal']

    for col in [x for x in tab_b.columns if x not in tab_b_new_cols]:
        repl = df_main.loc[df_main['Journal'].isin(tab_b['Journal']), ['Journal', col]]
        repl = repl.loc[repl[col].isna(), 'Journal']
        df_main.loc[df_main['Journal'].isin(repl), col] = tab_b.loc[repl, col].values

    return df_main


def drop_if_too_many_bad_segmentation(
        labels: np.array,
        max_share_bad_segmentation: float,
        ) -> np.array:
    # Remove bad segmentation cases if `max_share_bad_segmentation` is exceeded
    bad_segmentation_idxs = np.where(labels[:, 1] == 'bad cpd')[0]
    other_idxs = np.where(labels[:, 1] != 'bad cpd')[0]

    if len(bad_segmentation_idxs) / len(labels) <= max_share_bad_segmentation:
        # not too many in first place
        return labels

    np.random.seed(seed=42)
    nb_to_keep = int(max_share_bad_segmentation * len(other_idxs) / (1 - max_share_bad_segmentation))
    bad_cpd_idxs_to_keep = np.random.choice(bad_segmentation_idxs, size=nb_to_keep, replace=False)
    labels = labels[list(bad_cpd_idxs_to_keep) + list(other_idxs)]

    return labels


def init_verifiers() -> Dict[str, Callable]:
    verifiers = Verifiers()
    lookup_verify = {
        **{f'weight-{x}-mo': verifiers.verify_weight for x in (0, 1, 2, 3, 4, 6, 9, 12)},
        **{f'length-{x}-mo': verifiers.verify_length for x in (0, 12)},
        **{f'date-{x}-mo': verifiers.verify_date for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c1-{x}-mo': verifiers.verify_tab_b_123 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c2-{x}-mo': verifiers.verify_tab_b_123 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c3-{x}-mo': verifiers.verify_tab_b_123 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c4-{x}-mo': verifiers.verify_tab_b_123 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c5-{x}-mo': verifiers.verify_tab_b_int for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c6-{x}-mo': verifiers.verify_tab_b_int for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c7-{x}-mo': verifiers.verify_tab_b_12 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c8-{x}-mo': verifiers.verify_tab_b_123 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c9-{x}-mo': verifiers.verify_tab_b_12 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c10-{x}-mo': verifiers.verify_tab_b_12 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c11-{x}-mo': verifiers.verify_tab_b_12 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c12-{x}-mo': verifiers.verify_tab_b_12 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c13-{x}-mo': verifiers.verify_tab_b_12 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c14-{x}-mo': verifiers.verify_tab_b_12 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c15-{x}-mo': verifiers.verify_tab_b_123 for x in (1, 2, 3, 4, 6, 9, 12)},
        **{f'tab-b-c16-{x}-mo': verifiers.verify_tab_b_int for x in (1, 2, 3, 4, 6, 9, 12)},
        'dura-any-breastfeed': verifiers.verify_bfdurany,
        'preterm-birth': verifiers.verify_tab_b_12,
        'preterm-birth-weeks': verifiers.verify_tab_b_int, # FIXME prob cast 0 to 0=Mangler here!!
        'breastfeed-7-do': verifiers.verify_tab_b_123,
        **{f'nurse-name-{x}': verifiers.verify_nurse_name for x in (1, 2, 3)},
        }

    return lookup_verify


def gen_labels(
        labels_root: str,
        share_test: float,
        handle_bad_segmentation: str = 'keep',
        max_share_bad_segmentation: float = 1.0,
        ):
    """
    Generate label files based on transcribed data. The aim of this function is
    to serve as a final function to generate labels.

    Parameters
    ----------
    labels_root : str
        The directory where the label files are exported to.
    share_test : float
        Share of observations sorted away to test set.
    handle_bad_segmentation : str
        bad segmentation handling. One of "keep", "ignore", "drop". Default
        is "keep".
    max_share_bad_segmentation : float in [0, 1]
        How many labels (as a share of labels for a field) to at most allow to
        be "bad segmentation" cases. If too many, drop "bad segmentation"
        cases from labels until `max_share_for_bad_segmentation` is reached.

    Returns
    -------
    None.

    """

    # CONSTANTS: Could be made arguments...
    fn_df_main = 'Y:/RegionH/SPJ/Database/export_181106.txt'
    fn_map_lookup_df = 'Y:/RegionH/Scripts/users/tsdj/storage/maps/map_lookup_df.pkl'
    fn_df_nurse_names = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names.csv'
    fn_df_nurse_names_new = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names_additional_new_lise_data.csv'
    fn_df_tab_b_new = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\tab_b_additional_new_lise_data.csv'

    if handle_bad_segmentation not in ('keep', 'ignore', 'drop'):
        raise ValueError(f'handle_bad_segmentation must be one of (keep, ignore, drop), got {handle_bad_segmentation}')

    # Unless ignore bad segmentation, add them. If drop, we need them now
    # before we can later drop based on them
    add_bad_segmentation = handle_bad_segmentation != 'ignore' # NOTE does nothing as no new method implemented to add. Could based on metric obtained during segmentation

    if not isinstance(max_share_bad_segmentation, float):
        raise TypeError(f'max_share_bad_segmentation must be of type float, got {max_share_bad_segmentation} of type {type(max_share_bad_segmentation)}')

    if not 0 <= max_share_bad_segmentation <= 1:
        raise ValueError(f'max_share_bad_segmentation must be in [0, 1], got {max_share_bad_segmentation}')

    # Load datasets
    df_main = pd.read_table(fn_df_main, sep=';')
    df_main = df_main.drop_duplicates('Filename')

    # As identifier use journal name without file suffix and add image filename
    df_main['Journal'] = df_main['Filename'].transform(lambda x: x.split('.')[0])
    df_main['fname'] = df_main['Journal'] + '.jpg'

    with open(fn_map_lookup_df, 'rb') as file:
        map_lookup_df = pickle.load(file)

    empty_cells = load_mod_weight()

    # Merge info on nurse names into main_df
    df_main = merge_nurse_names(
        df_main=df_main,
        fn_df_nurse_names=fn_df_nurse_names,
        fn_df_nurse_names_new=fn_df_nurse_names_new,
        )

    # Merge info on Table B from new EPI data dump
    df_main = merge_new_table_b(
        df_main=df_main,
        fn_df_tab_b_new=fn_df_tab_b_new,
        map_lookup_df=map_lookup_df,
        )

    lookup_verify = init_verifiers()

    for i, (key, value) in enumerate(map_lookup_df.items(), start=1):
        if value not in df_main.columns:
            warnings.warn(f'Skipping {key} as column not found')
            continue

        print(f'Creating/appending labels for/to {key} ({i}/{len(map_lookup_df)})')
        sub = df_main[['fname', 'Journal', value]].copy()

        # Handle empty (this is only for weight, see `load_mod_weight`)
        idxs_empty = sub['Journal'].isin(empty_cells.loc[empty_cells['cell'] == key, 'Journal'])
        sub.loc[idxs_empty, value] = 'empty'

        # Keep only where we have the label
        labels = sub[['fname', value]].dropna()

        # Verify labels meaningful - else cast to None
        labels[value] = labels[value].astype(str) # some values are float, cast str
        labels[value] = list(map(lookup_verify[key], labels[value]))
        labels = labels[['fname', value]].dropna() # Drop bad cases

        if handle_bad_segmentation == 'drop':
            labels = labels[labels[value] != 'bad cpd']

        fn_out_train = os.path.join(labels_root, 'train', f'{key}.npy')
        fn_out_test = os.path.join(labels_root, 'test', f'{key}.npy')

        # Check if exist, if they do load and then append to them
        if os.path.isfile(fn_out_train):
            labels_train = np.load(fn_out_train, allow_pickle=True)
        else:
            labels_train = np.empty(shape=(0, 2))

        if os.path.isfile(fn_out_test):
            labels_test = np.load(fn_out_test, allow_pickle=True)
        else:
            labels_test = np.empty(shape=(0, 2))

        new_labels = labels[~labels['fname'].isin(np.concatenate([labels_train[:, 0], labels_test[:, 0]]))]
        new_labels = np.array(new_labels)
        new_labels = drop_if_too_many_bad_segmentation(
            new_labels,
            max_share_bad_segmentation,
            )

        if len(new_labels) == 0: # no new labels, continue to next field
            print(f'No new labels to append to {key}')
            continue

        if len(new_labels) == 1: # Not possible to split, so only add to train
            labels_train_new = new_labels
            labels_test_new = np.empty(shape=(0, 2))
        else:
            labels_train_new, labels_test_new = train_test_split(
                new_labels,
                test_size=share_test,
                random_state=42,
                )

        labels_train_new = np.concatenate([labels_train, labels_train_new])
        labels_test_new = np.concatenate([labels_test, labels_test_new])

        np.save(fn_out_train, labels_train_new)
        np.save(fn_out_test, labels_test_new)

    # Below not alway work, e.g., 'tab-b-c9-1-mo' not exist for `handle_bad_cpd='ignore'`
    # coolstuff = {x: np.load(
    #     ''.join((labels_root, 'train/', x, '.npy')), allow_pickle=True,
    #     ) for x in map_lookup_df.keys()}
    # print({k: len(v) for k, v in coolstuff.items()})
    # print({k: len(set(v[:, 1])) for k, v in coolstuff.items()})
    # print({k: sum(v[:, 1] == 'bad cpd') for k, v in coolstuff.items()})

    # Maybe also use this script to generate auxillary columns, as this allows
    # us to use 'date-0-mo', for example (derive from the birth date cols or
    # even CPR).
    # Prob do in other script run BEFORE this, so it can be incorporated
    # directly in `gen_map_lookup_df.py` as well!!


if __name__ == '__main__':
    gen_labels(
        labels_root=r'Y:\RegionH\Scripts\data\storage\labels\keep',
        share_test=0.1,
        handle_bad_segmentation='keep',
        max_share_bad_segmentation=0.1,
        )
