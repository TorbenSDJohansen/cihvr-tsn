# -*- coding: utf-8 -*-
"""
@author: tsdj

"""


import os
import pickle
import json
import time
import string

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def _load_mod_weight(add_bad_cpd: bool):
    map_raw_to_final = {'0': 'empty', 0: 'empty', '1': 'bad cpd', 1: 'bad cpd'}

    with open('Y:/RegionH/Scripts/users/cmd/BW_low_confidence5200.json', 'rb') as file:
        entires_new = json.load(file)

    entries = entires_new['elements'][:entires_new['cursor']]
    new_label_info = {'fname': [], 'cell': [], 'label': []}

    for entry in entries:
        label_raw = entry['properties']['Birth Weight']

        if not label_raw in map_raw_to_final.keys():
            print(entry)
            continue

        new_label_info['fname'].append(entry['name'])
        new_label_info['cell'].append(entry['folder'].split('\\')[-1])
        new_label_info['label'].append(map_raw_to_final[label_raw])

    new_label_info = pd.DataFrame(new_label_info)

    bad_cpd_files = set(new_label_info[new_label_info['label'] == 'bad cpd']['fname'])

    # check if any cell is implicitly double labelled, i.e. both empty and
    # bad cpd!
    nli_empty = new_label_info[new_label_info['label'] == 'empty']
    idxs_to_change = nli_empty[nli_empty['fname'].isin(bad_cpd_files)].index

    if add_bad_cpd:
        new_label_info.loc[idxs_to_change, 'label'] = 'bad cpd'
        assert new_label_info[new_label_info['label'] == 'empty']['fname'].isin(bad_cpd_files).sum() == 0 # pylint: disable=C0301

    empty_cells = new_label_info[new_label_info['label'] == 'empty']

    return empty_cells, bad_cpd_files


def _get_mat_summary(tm): # pylint: disable=C0103
    if tm is None:
        return None, None, None

    det = np.linalg.det(tm)

    rotx, roty = tm[:2, 2]

    return det, rotx, roty


def _construct_df(log: dict) -> pd.DataFrame:
    data = []

    for key, value in log.items():
        tm = value['transform-matrix'] if 'transform-matrix' in value.keys() else None # pylint: disable=C0103
        det, rotx, roty = _get_mat_summary(tm)
        nx, ny = value['nx'], value['ny']  # pylint: disable=C0103

        data.append((
            key, value['succesful'],
            nx, ny,
            det, rotx, roty,
            ))

    columns = ['page', 'succesful', 'nx', 'ny', 'det', 'rotx', 'roty']

    data = pd.DataFrame(data, columns=columns)

    return data


class Verifiers: # pylint: disable=C0115
    _allowed_chars_names = set(list(string.ascii_lowercase) + ['æ', 'ø', 'å'])

    @staticmethod
    def verify_weight(weight): # pylint: disable=C0116
        if weight in ('bad cpd', 'empty'):
            return weight

        _allowed_range = set(range(1000, 20_000)) # Maybe change?

        if int(float(weight)) not in _allowed_range:
            print(f'Bad weight value: {weight}. Casting to None.')
            return None

        return weight

    @staticmethod
    def verify_length(length): # pylint: disable=C0116
        if length == 'bad cpd':
            return length

        _allowed_range = set(range(20, 100)) # Maybe change?

        if int(float(length)) not in _allowed_range:
            print(f'Bad length value: {length}. Casting to None.')
            return None

        return length

    @staticmethod
    def verify_date(date): # pylint: disable=C0116
        if date in ('bad cpd', ',:,:,'):
            return date

        try:
            time.strptime(':'.join(date.split(':')[:2]), '%d:%m')
        except ValueError:
            print(f'Bad date value: {date}. Casting to None.')

        return date

    @staticmethod
    def verify_tab_b_123(tab_b_entry): # pylint: disable=C0116
        if tab_b_entry in ('bad cpd', '0=Mangler'):
            return tab_b_entry

        _allowed = {1, 2, 3}

        if int(tab_b_entry[0]) not in _allowed:
            print(f'Bad table B (1, 2, 3) value: {tab_b_entry}. Casting to None.')
            return None

        return tab_b_entry

    @staticmethod
    def verify_tab_b_12(tab_b_entry): # pylint: disable=C0116
        if tab_b_entry in ('bad cpd', '0=Mangler'):
            return tab_b_entry

        _allowed = {1, 2}

        if int(tab_b_entry[0]) not in _allowed:
            print(f'Bad table B (1, 2) value: {tab_b_entry}. Casting to None.')
            return None

        return tab_b_entry

    @staticmethod
    def verify_tab_b_int(tab_b_entry): # pylint: disable=C0116
        if tab_b_entry in ('bad cpd', '0=Mangler'):
            return tab_b_entry

        _allowed = set(range(24))

        if int(float(tab_b_entry)) not in _allowed:
            print(f'Bad table B (int) value: {tab_b_entry}. Casting to None.')
            return None

        return tab_b_entry

    @staticmethod
    def verify_bfdurany(duration): # pylint: disable=C0116
        if duration == 'bad cpd':
            return duration

        _allowed = set(range(14)) # from "tasteinstruktion"

        if int(float(duration)) not in _allowed:
            print(f'Bad duration value: {duration}. Casting to None.')
            return None

        return duration

    def verify_nurse_name(self, name: str): # pylint: disable=C0116
        if name in ('bad cpd', '0=Mangler'):
            return name

        for subname in name.split():
            if not set(subname).issubset(self._allowed_chars_names):
                print(f'Bad nurse name: {name}. Casting to None.')
                return None

        return name


def load_100x112_tab_b_sample_filenames():
    folder = r'Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined\labels\keep-tab-b-test\test'
    files = [os.path.join(folder, x) for x in os.listdir(folder)]
    labels_test_100x112 = np.concatenate([np.load(x, allow_pickle=True) for x in files])
    files_test_100x112 = [x.split('.page-')[0] for x in labels_test_100x112[:, 0]]
    files_test_100x112 = set(files_test_100x112)

    assert len(files_test_100x112) == 100

    return files_test_100x112


def gen_labels(
        labels_root: str,
        share_test: float,
        log_file: str,
        handle_bad_cpd: str = 'ignore',
        ):
    """
    Generate label files based on transcribed data. The aim of this function is
    to serve as a final function to generate labels. It is built after the cmd-
    tsdj merge project.

    Parameters
    ----------
    labels_root : str
        The directory where the label files are exported to.
    share_test : float
        Share of observations sorted away to test set.
    log_file : str
        Log file used for CPD. Includes info useful to identify bad CPD cases
        etc.
    handle_bad_cpd : str
        bad cpd handling. One of "keep", "ignore", "drop". Default is "ignore".

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

    # TODO *NEED* to re-run everything once new segmentations, as the labels
    # now "bad cpd" might no longer be -- and in principle some currently fine
    # labels will need to be changed to failed segmentation

    assert handle_bad_cpd in ('keep', 'ignore', 'drop')
    # Unless ignore bad cpd, add them. If drop, we need them now before we can
    # later drop based on them
    add_bad_cpd = handle_bad_cpd != 'ignore'

    df_main = pd.read_table(fn_df_main, sep=';')
    df_main = df_main.drop_duplicates('Filename')

    with open(fn_map_lookup_df, 'rb') as file:
        map_lookup_df = pickle.load(file)

    with open(log_file, 'rb') as file:
        log = pickle.load(file)

    empty_cells, bad_cpd_files = _load_mod_weight(add_bad_cpd)

    log_df = _construct_df(log)
    log_df = log_df[log_df['succesful']]
    log_df['Filename'] = log_df['page'].str.split('.page').apply(
        lambda x: x[0],
        )

    # Merge info on nurse names into main_df
    nurse_names = pd.read_csv(fn_df_nurse_names)
    nurse_names_new = pd.read_csv(fn_df_nurse_names_new)

    assert set(nurse_names['Filename']).intersection(nurse_names_new['Filename']) == set()

    nurse_names = pd.concat([
        nurse_names[['Filename', 'nurse-name-1', 'nurse-name-2', 'nurse-name-3']],
        nurse_names_new[['Filename', 'nurse-name-1', 'nurse-name-2', 'nurse-name-3']],
        ])
    df_main = df_main.merge(nurse_names, on='Filename', how='left')

    # Merge info on Table B from new Lise data.
    tab_b = pd.read_csv(fn_df_tab_b_new)

    # Drop all the 1-month columns as suggested by Lise
    bad_cols = [map_lookup_df[f'tab-b-c{x}-1-mo'] for x in range(1, 17)]
    tab_b = tab_b.drop(columns=bad_cols)

    # Drop all rows corresponding to the manual 100x112 test sample
    files_in_tab_b_100x112_test = load_100x112_tab_b_sample_filenames()
    tab_b = tab_b[~tab_b['Filename'].isin(files_in_tab_b_100x112_test)]

    # Identify which columns are not already in df_main
    tab_b_new_cols = ['Filename'] + [x for x in tab_b.columns if x not in df_main.columns]

    # Safe with normal merge for cols not already in df_main
    df_main = df_main.merge(tab_b[tab_b_new_cols], on='Filename', how='left')

    # For those already in df_main, only replace when no value. Useful to let
    # `tab_b` index be filename to match filenames between the two data frames
    tab_b.index = tab_b['Filename']

    for col in [x for x in tab_b.columns if x not in tab_b_new_cols]:
        repl = df_main.loc[df_main['Filename'].isin(tab_b['Filename']), ['Filename', col]]
        repl = repl.loc[repl[col].isna(), 'Filename']
        df_main.loc[df_main['Filename'].isin(repl), col] = tab_b.loc[repl, col].values

    # The below "clones" whenever multiple pages present. Note, however, that
    # the page is sometimes missing -> as is the case due to unsuccesful
    # cropping.
    merged = df_main.merge(log_df, how='left', on='Filename')
    merged.loc[merged['page'].isna(), 'page'] = merged['Filename'] + '.page-UNKNOWN'

    # Check manual labelling of empty and bad CPD matches in format.
    assert set(empty_cells['fname']).issubset(set(merged['page']))
    assert bad_cpd_files.issubset(set(merged['page']))

    if add_bad_cpd:
        # Handle bad cpd (manual checks)
        idxs_bad_cpd1 = merged['page'].isin(bad_cpd_files)
        merged.loc[idxs_bad_cpd1, list(map_lookup_df.values())] = 'bad cpd'

        # Handle bad CPD based on matrix determinant.
        det_ut = 1.03 # Maybe change?
        det_lt = 0.91 # Maybe change?
        idxs_bad_cpd2 = ~((merged['det'] <= det_ut) & (merged['det'] >= det_lt))
        merged.loc[idxs_bad_cpd2, list(map_lookup_df.values())] = 'bad cpd'

    # Potentially more CPD checks...

    verifiers = Verifiers()
    lookup_verify = {
        'weight-0-mo': verifiers.verify_weight,
        'weight-1-mo': verifiers.verify_weight,
        'weight-2-mo': verifiers.verify_weight,
        'weight-3-mo': verifiers.verify_weight,
        'weight-4-mo': verifiers.verify_weight,
        'weight-6-mo': verifiers.verify_weight,
        'weight-9-mo': verifiers.verify_weight,
        'weight-12-mo': verifiers.verify_weight,
        'length-0-mo': verifiers.verify_length,
        'length-12-mo': verifiers.verify_length,
        'date-1-mo': verifiers.verify_date,
        'date-2-mo': verifiers.verify_date,
        'date-3-mo': verifiers.verify_date,
        'date-4-mo': verifiers.verify_date,
        'date-6-mo': verifiers.verify_date,
        'date-9-mo': verifiers.verify_date,
        'date-12-mo': verifiers.verify_date,
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
        'preterm-birth-weeks': verifiers.verify_tab_b_int,
        'breastfeed-7-do': verifiers.verify_tab_b_123,
        'nurse-name-1': verifiers.verify_nurse_name,
        'nurse-name-2': verifiers.verify_nurse_name,
        'nurse-name-3': verifiers.verify_nurse_name,
        }

    for i, (key, value) in enumerate(map_lookup_df.items(), start=1):
        if value not in merged.columns:
            # e.g. 'tab-b-c9-1-mo' when `handle_bad_cpd='ignore'` since then
            # `add_bad_cpd` is False and the column is never added to `merged`
            print(f'Skipping {key} as column not found')
            continue

        print(f'Creating/appending labels to {key} ({i}/{len(map_lookup_df)})')
        sub = merged[['page', 'det', value]].copy()

        # Handle empty (this is only for weight, see `_load_mod_weight`)
        idxs_empty = sub['page'].isin(empty_cells.loc[empty_cells['cell'] == key, 'fname'])
        sub.loc[idxs_empty, value] = 'empty'

        # Keep only where we have the label
        labels = sub[['page', value]].dropna()

        # Verify labels meaningful - else cast to None
        labels[value] = labels[value].astype(str) # some values are float, cast str
        labels[value] = list(map(lookup_verify[key], labels[value]))
        labels = labels[['page', value]].dropna() # Drop bad cases

        if handle_bad_cpd == 'drop':
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

        new_labels = labels[~labels['page'].isin(np.concatenate([labels_train[:, 0], labels_test[:, 0]]))]
        new_labels = np.array(new_labels)

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
                random_state=1,
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
    # Implement fourth method, which keeps label but also signals bad cpd; now
    # possible to predict the likely label as well as indicator for bad cpd
    gen_labels(
        labels_root='Y:/RegionH/Scripts/users/tsdj/storage/image-datasets-joined/labels/keep/',
        share_test=0.1,
        log_file='Y:/RegionH/Scripts/users/tsdj/storage/cpd-root/210304-tab-b-cmd-tsdj-merge/log-merged.pkl', # pylint: disable=C0301
        handle_bad_cpd='keep',
        )
    gen_labels(
        labels_root='Y:/RegionH/Scripts/users/tsdj/storage/image-datasets-joined/labels/ignore/',
        share_test=0.1,
        log_file='Y:/RegionH/Scripts/users/tsdj/storage/cpd-root/210304-tab-b-cmd-tsdj-merge/log-merged.pkl', # pylint: disable=C0301
        handle_bad_cpd='ignore',
        )
    gen_labels(
        labels_root='Y:/RegionH/Scripts/users/tsdj/storage/image-datasets-joined/labels/drop/',
        share_test=0.1,
        log_file='Y:/RegionH/Scripts/users/tsdj/storage/cpd-root/210304-tab-b-cmd-tsdj-merge/log-merged.pkl', # pylint: disable=C0301
        handle_bad_cpd='drop',
        )
