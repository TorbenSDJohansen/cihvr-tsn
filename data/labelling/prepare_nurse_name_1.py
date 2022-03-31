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
possible to read for tsdj. Perhaps let those be 1, quick to type and names
never contain numbers.

IMPORTANT TO CONSIDER: Worth to incorporate first name directly now. If ever
needed, much better to do now than first perform round only for last names.

Note that when using both first and last name, the criteria proposed probably
needs to be applies separately for each, and then take union.

"""


import os
import shutil

import json

import numpy as np
import pandas as pd


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


def _create_ens_workspace(
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
        ):
    pred_ln.columns = [x + '_ln' if x != 'filename_full' else x for x in pred_ln.columns]
    pred_fn.columns = [x + '_fn' if x != 'filename_full' else x for x in pred_fn.columns]

    pred = pred_ln.merge(pred_fn, on='filename_full', how='inner')
    assert len(pred) == len(pred_ln) == len(pred_fn)

    # Remove those already labelled
    if additional_labelled is not None:
        raise NotImplementedError

    pred['journal_name'] = pred['filename_full'].str.split('/').apply(
        lambda x: x[-1][:x[-1].lower().find('.pdf') + 4]
        )
    pred = pred[~pred['journal_name'].isin(labelled['Filename'])]

    # Matched or not? For now use unmatched, probably healthy for the `nb_each_unique` part
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

    # CONSIDER: best to do as below with bad cpd and empty (cast to ?) or remove such cases completely?
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
    elements = []

    for filename_full, pred_full in sub.values:
        _, filename = os.path.split(filename_full)
        filename_full_out = os.path.join(outdir, 'images', filename)

        if not os.path.isfile(filename_full_out):
            shutil.copy(filename_full, filename_full_out)

        elements.append({
            'name': filename,
            'folder': os.path.join(outdir, 'images'),
            'path': filename_full_out,
            'properties': {'name': pred_full}
            })

    wsp['elements'] = elements

    with open(os.path.join(outdir, 'wsp.json'), 'w') as wsp_file:
        json.dump(wsp, wsp_file)


def _workspace_to_labels():
    raise NotImplementedError


def main(current_round: int):
    r'''
    Take as input predictions of first and last name. Could be matched, could
    also be raw.
    Remember to drop those already labelled.
    Select proper subset (see notes above).
    Prepare format for ens app, see json.load(open(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-tab-b\Torben_tab_b.json', 'r'))
    Then label manually with these as initial predictions.
    Finally map to label format.
    '''

    if current_round == 1:
        # ROUND 1
        fn_pred_ln=r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\tl-lr-0.25\preds_matched.csv'
        fn_pred_fn=r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\tl-lr-0.0625\preds_matched.csv'

        pred_ln = pd.read_csv(fn_pred_ln)
        pred_fn = pd.read_csv(fn_pred_fn)
        labelled = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse_names.csv')
        additional_labelled = None

        outdir = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-nurse-name-1\round-1'

        nb_each_unique = 1 # Useful, when matching
        nb_prob_based = 0 #
        nb_random_based = 0
        nb_christiansen = 0
        use_matching = True
    elif current_round == 2:
        # Perhaps begin to use matching?
        raise NotImplementedError
    else:
        raise Exception

    _create_ens_workspace(
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
        )

    # FIXME/TODO at analysis-stage, drop all nurses that occur rarely. These
    # are more likely to be incorrect predictions and they are also more or
    # less useful for downstream analyses

if __name__ == '__main__':
    main(current_round=1)
