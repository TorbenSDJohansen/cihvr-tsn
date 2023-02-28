# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Need to label more nurse names to achieve sufficiently high accuracy. Strategic
considerations and notes below:

PROBLEM: Almost impossible to read many of the names. Direct labelling not
possible to perform by tsdj.

APPROACH: Use first round predictions. Based on these, select subset of those
that is NOT part of the already labelled data. Base this subset on, e.g., some
of the following criteria:
    1) Low confidence
    2) Not 0=Mangler or bad cpd
    3) k-for-each-unique prediction
Then label, but be aware that many will need to be skipped due to not being
possible to read for tsdj. Perhaps let those be x, quick to type and never the
correct name.

IMPORTANT TO CONSIDER: Worth to incorporate first name directly now. If ever
needed, much better to do now than first perform round only for last names.

Note that when using both first and last name, the criteria proposed probably
needs to be applied separately for each, and then take union.

"""


import argparse
import os
import shutil
import math
import string
import warnings

import json

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--round', type=int)
    parser.add_argument('--task', type=str, choices=['create-workspaces', 'create-labels'], default='create-workspaces')

    args = parser.parse_args()

    if args.round < 0:
        raise ValueError(f'--round cannot be negative, got {args.round}')


    return args


def _select_up_to_k_random(series: pd.Series, nb_each: int):
    nb_each = min(nb_each, len(series))

    return np.random.choice(series.index, size=nb_each, replace=False)


def _select_rows(
        pred: pd.DataFrame,
        suffix: str,
        nb_each_unique: int,
        nb_prob_based: int,
        nb_random_based: int,
        nb_christiansen: int,
        ):
    pred_col = 'pred' + suffix
    prob_col = 'prob' + suffix if suffix[:2] != '_m' else 'prob' + suffix[2:]

    # Not interested in labelled bad cpd or empty, as these are easy
    sub = pred[~pred[pred_col].isin(('bad cpd', '0=Mangler'))]

    selected = []
    np.random.seed(42)

    # Select up to k of each unique prediction (useful to get diverse)
    # Fragile when not matching (many uniques from, e.g., empty are selected)
    if nb_each_unique > 0:
        k_of_each = sub.groupby(pred_col).apply(lambda x: _select_up_to_k_random(x, nb_each_unique))
        k_of_each = [item for value in k_of_each for item in value]
        assert len(k_of_each) == len(set(k_of_each))
        selected.extend(k_of_each)

    # Select rows where more uncertain with higher probability
    if nb_prob_based > 0:
        prob_select = (sub[prob_col].min() + 0.001) / (sub[prob_col].values + 0.001)
        random_weighted = np.random.choice(sub.index, size=nb_prob_based, replace=False, p=prob_select / prob_select.sum())
        selected.extend(list(random_weighted))

    # Select rows with uniform probability
    if nb_random_based > 0:
        random_uniform = np.random.choice(sub.index, size=nb_random_based, replace=False)
        selected.extend(list(random_uniform))

    # Select christiansen cases to learn not to cast empty to this
    if suffix[-2:] == 'ln' and nb_christiansen > 0:
        random_christiansen = np.random.choice(
            sub[sub[pred_col] == 'christiansen'].index, size=nb_christiansen, replace=False,
            )
        selected.extend(list(random_christiansen))

    # selected likely contains duplicates (since multiple inclusion criteria).
    # Now drop to unique
    selected = list(set(selected))

    return selected


def _create_ens_workspaces(
        pred_ln: pd.DataFrame,
        pred_fn: pd.DataFrame,
        labelled: pd.DataFrame,
        additional_labelled: pd.DataFrame,
        outdir: str,
        nb_each_unique: int,
        nb_prob_based: int,
        nb_random_based: int,
        nb_christiansen: int,
        use_matching: bool,
        nb_packages: int,
        ):
    pred_ln.columns = [x + '_ln' if x != 'filename_full' else x for x in pred_ln.columns]
    pred_fn.columns = [x + '_fn' if x != 'filename_full' else x for x in pred_fn.columns]

    pred = pred_ln.merge(pred_fn, on='filename_full', how='inner')
    assert len(pred) == len(pred_ln) == len(pred_fn)

    if additional_labelled is not None:
        raise NotImplementedError

    pred['journal_name'] = pred['filename_full'].str.split('/').apply(
        lambda x: x[-1][:x[-1].lower().find('.pdf') + 4]
        )
    pred = pred[~pred['journal_name'].isin(labelled['Filename'])]

    idxs_ln = _select_rows(
        pred,
        suffix='_m_ln' if use_matching else '_ln',
        nb_each_unique=nb_each_unique,
        nb_prob_based=nb_prob_based,
        nb_random_based=nb_random_based,
        nb_christiansen=nb_christiansen,
        )
    idxs_fn = _select_rows(
        pred,
        suffix='_m_fn' if use_matching else '_fn',
        nb_each_unique=nb_each_unique,
        nb_prob_based=nb_prob_based,
        nb_random_based=nb_random_based,
        nb_christiansen=nb_christiansen,
        )

    # Select union (likely to be large overlap)
    sub = pred[pred.index.isin(set(idxs_ln + idxs_fn))].copy()

    # CONSIDER: Best to do as below with bad cpd and empty (cast to ?) or remove such cases completely?
    if use_matching:
        sub['pred_m_fn'] = sub['pred_m_fn'].replace({'bad cpd': '?', '0=Mangler': '?'})
        sub['pred_m_ln'] = sub['pred_m_ln'].replace({'bad cpd': '?', '0=Mangler': '?'})
        sub['pred_full'] = sub['pred_m_fn'] + ' ' + sub['pred_m_ln']
    else:
        sub['pred_fn'] = sub['pred_fn'].replace({'bad cpd': '?', '0=Mangler': '?'})
        sub['pred_ln'] = sub['pred_ln'].replace({'bad cpd': '?', '0=Mangler': '?'})
        sub['pred_full'] = sub['pred_fn'] + ' ' + sub['pred_ln']

    sub = sub[['filename_full', 'pred_full']]

    wsp = {
        'tags': ['name'],
        'currentTag': 'default_tag',
        'isModified': False,
        'savePath': None,
        'cursor': 1,
        'useInference': True,
        }

    offset = math.ceil(len(sub) / nb_packages)

    for i in range(nb_packages):
        outdir_package = os.path.join(outdir, f'package-{i + 1}')
        elements = []

        if not os.path.isdir(outdir_package):
            os.makedirs(outdir_package)
            os.makedirs(os.path.join(outdir_package, 'images'))

        for filename_full, pred_full in sub.values[i * offset: (i + 1) * offset]:
            _, filename = os.path.split(filename_full)
            filename_full_out = os.path.join(outdir_package, 'images', filename)

            if not os.path.isfile(filename_full_out):
                shutil.copy(filename_full, filename_full_out)

            elements.append({
                'name': filename,
                'folder': os.path.join(outdir_package, 'images'),
                'path': filename_full_out,
                'properties': {'name': pred_full},
                })

        wsp['elements'] = elements # re-use and overwrite this part

        with open(os.path.join(outdir_package, 'wsp.json'), 'w') as wsp_file:
            json.dump(wsp, wsp_file)


def load_workspace(file: str) -> pd.DataFrame:
    with open(file, 'r', encoding='utf-8') as stream:
        workspace = json.load(stream)

    cursor = workspace['cursor'] + 1 # image reached in workspace
    workspace = workspace['elements']

    if cursor != len(workspace):
        raise ValueError(f'workspace {file} not appeaer to gone all trough, cursor = {cursor} != {len(workspace)} = len(workspace)')

    workspace = workspace[:cursor] # only keep images reached

    labels = pd.DataFrame({
        'image_id':[x['name'] for x in workspace], # '0024.jpg'
        # 'file': [x['path'] for x in workspace], # 'path/to/0024.jpg'
        'label': [x['properties']['name'] for x in workspace], # 'torben johansen'
        })

    return labels


def load_labels(file: str) -> pd.DataFrame:
    if file.endswith('.npy'):
        labels = np.load(file, allow_pickle=True)
        labels = pd.DataFrame(labels, columns=['image_id', 'label'])
    else:
        labels = pd.read_csv(file)

    return labels


def save_labels(file: str, labels: pd.DataFrame):
    labels = labels.values
    np.save(file, labels, allow_pickle=True)


def recast_chars(name: str):
    name = name.replace('.', ' ')
    name = name.replace('-', '')
    name = name.replace('ü', 'u')
    name = name.strip()
    name = name.lower()

    return name


def check_name_valid(name: str):
    allowed_chars = set(list(string.ascii_lowercase) + ['æ', 'ø', 'å', ' '])

    for char in name:
        if char not in allowed_chars:
            return False

    return True


def _workspace_to_labels(
        wsp_dir: str,
        labels_file_train: str,
        labels_file_test: str,
        ):
    if not os.path.isdir(wsp_dir):
        raise NotADirectoryError(f'workspace directory {wsp_dir} does not exist')

    wsp_files = [os.path.join(wsp_dir, x) for x in os.listdir(wsp_dir)]
    wsp_files = [x for x in wsp_files if x.endswith('.json')]

    if len(wsp_files) == 0:
        raise FileNotFoundError(f'no workspaces found in {wsp_dir}')

    print(f'creating labels from {len(wsp_files)} workspaces')

    labels = []

    for file in wsp_files:
        labels.append(load_workspace(file))

    labels = pd.concat(labels)
    labels['label'] = labels['label'].apply(recast_chars)

    new_labels = labels[labels['label'] != 'x'].copy() # drop those failed to label

    is_invalid = ~new_labels['label'].apply(check_name_valid)

    if is_invalid.sum() > 0:
        warnings.warn(f'dropping {is_invalid.sum()} invalid names:\n {new_labels[is_invalid]}')

    new_labels = new_labels[~is_invalid]
    new_labels['label'] = new_labels['label'].replace({'': '0=Mangler'})

    old_labels_train = load_labels(labels_file_train)
    old_labels_test = load_labels(labels_file_test)

    # Check no new labels already in current labels
    assert set(old_labels_train['image_id']).union(old_labels_test['image_id']).intersection(labels['image_id']) == set()

    share_train = len(old_labels_train) / (len(old_labels_test) + len(old_labels_train))
    new_labels_train, new_labels_test = train_test_split(
        new_labels,
        train_size=share_train,
        random_state=42,
        )

    new_labels_train = pd.concat([old_labels_train, new_labels_train])
    new_labels_test = pd.concat([old_labels_test, new_labels_test])

    save_labels(labels_file_train, new_labels_train)
    save_labels(labels_file_test, new_labels_test)


def create_workspaces(current_round: int):
    r'''
    Take as input predictions of first and last name. Could be matched, could
    also be raw.
    Remember to drop those already labelled.
    Select proper subset (see notes above).
    Prepare format for ens app, see json.load(open(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-tab-b\Torben_tab_b.json', 'r'))
    Then label manually with these as initial predictions.
    Finally map to label format.
    '''

    if current_round == 0: # Test round to MW
        fn_pred_ln=r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\tl-lr-0.25\preds_matched.csv'
        fn_pred_fn=r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\tl-lr-0.0625\preds_matched.csv'

        pred_ln = pd.read_csv(fn_pred_ln)
        pred_fn = pd.read_csv(fn_pred_fn)
        labelled = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names.csv')
        additional_labelled = None

        outdir = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-nurse-name-1\round-0'

        nb_each_unique = 1 # Useful, when matching
        nb_prob_based = 0 #
        nb_random_based = 0
        nb_christiansen = 0
        use_matching = True
        nb_packages = 1
    elif current_round == 1: # To Malthe Hauschildt Veje <malthehv@econ.ku.dk>
        fn_pred_ln=r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\tl-lr-0.25\preds_matched.csv'
        fn_pred_fn=r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\tl-lr-0.0625\preds_matched.csv'

        pred_ln = pd.read_csv(fn_pred_ln)
        pred_fn = pd.read_csv(fn_pred_fn)
        labelled_1 = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names.csv')
        labelled_2 = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names_additional_new_lise_data.csv')
        labelled = pd.concat([labelled_1, labelled_2])
        additional_labelled = None

        outdir = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-nurse-name-1\round-1'

        nb_each_unique = 5 # Useful, when matching
        nb_prob_based = 1000
        nb_random_based = 4000
        nb_christiansen = 0
        use_matching = True
        nb_packages = 30
    elif current_round == 2:
        raise NotImplementedError

        fn_pred_ln=None
        fn_pred_fn=None

        pred_ln = pd.read_csv(fn_pred_ln)
        pred_fn = pd.read_csv(fn_pred_fn)
        labelled_1 = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names.csv')
        labelled_2 = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names_additional_new_lise_data.csv')
        labelled = pd.concat([labelled_1, labelled_2])
        additional_labelled = None # FIXME add

        outdir = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-nurse-name-1\round-2'

        # TODO change vals perhaps
        nb_each_unique = 5 # Useful, when matching
        nb_prob_based = 1000
        nb_random_based = 4000
        nb_christiansen = 0
        use_matching = True
        nb_packages = 30
    else:
        raise Exception

    _create_ens_workspaces(
        pred_ln=pred_ln,
        pred_fn=pred_fn,
        labelled=labelled,
        outdir=outdir,
        additional_labelled=additional_labelled, # TODO once we add to original
        nb_each_unique=nb_each_unique,
        nb_prob_based=nb_prob_based,
        nb_random_based=nb_random_based,
        nb_christiansen=nb_christiansen,
        use_matching=use_matching,
        nb_packages=nb_packages,
        )

    # FIXME/TODO at analysis-stage, drop all nurses that occur rarely. These
    # are more likely to be incorrect predictions and they are also more or
    # less useless for downstream analyses


def create_labels(current_round: int):
    if current_round == 0:
        raise ValueError('no 0th round for task create-labels')

    if current_round == 1: # returned from Malthe Hauschildt Veje <malthehv@econ.ku.dk>
        wsp_dir = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-nurse-name-1\round-1-returned'

        _labels_dir = r'Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined\labels-restrict-share-bad-cpd\keep'
        labels_file_train = os.path.join(_labels_dir, r'train\nurse-name-1.npy')
        labels_file_test = os.path.join(_labels_dir, r'test\nurse-name-1.npy')
    else:
        raise Exception

    if not os.path.isfile(labels_file_train):
        raise FileNotFoundError(f'no file {labels_file_train}')
    if not os.path.isfile(labels_file_test):
        raise FileNotFoundError(f'no file {labels_file_test}')

    _workspace_to_labels(
        wsp_dir=wsp_dir,
        labels_file_train=labels_file_train,
        labels_file_test=labels_file_test,
        )


def main():
    args = parse_args()

    if args.task == 'create-workspaces':
        create_workspaces(args.round)
    elif args.task == 'create-labels':
        create_labels(args.round)


if __name__ == '__main__':
    main()
