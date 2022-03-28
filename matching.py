# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import time
import difflib
import argparse
import os
import pickle

import numpy as np
import pandas as pd


class MatchToStr():
    """
    Matches strings to a set of valid strings - such as names.

    Parameters
    ----------
    potential_strs : set
        The valid strings to be matched to.
    cutoff : float
        The lower threshold to performin matching. Must be in [0, 1]. Default
        is 0.6.
    ignore : set
        Cases of `strs_to_match` to ignore, i.e. not perform matching on.
        Relevant to not match e.g. bad cpd, 0=Mangler, etc. Default is None, in
        which case the empty set is used, i.e. no such cases.

    Returns
    -------
    None.

    """
    def __init__(self, potential_strs: set, cutoff: float = 0.6, ignore: set = None):
        if ignore is None:
            ignore = set()

        assert isinstance(potential_strs, set)
        assert 0 <= cutoff <= 1
        assert isinstance(ignore, set)

        self.potential_strs = potential_strs
        self.cutoff = cutoff
        self.ignore = ignore
        self.fuzzy_map = dict()

    def match(self, strs_to_match: np.ndarray) -> (np.ndarray, int, int, dict):
        """
        Matches strings to a set of valid strings - such as names.

        Does not perform matching for strings in `ignore`. Returns UNMATCHABLE
        if match "similarity" is below `cutoff`.

        Parameters
        ----------
        strs_to_match : np.ndarray
            The strings to match agains `self.potential_strs`.

        Returns
        -------
        strs_matched : np.ndarray
            The modified input, where all strings not in `self.potential_strs`
            have been matched to the nearest valid string.
        nb_exact : int
            The number of exact matches perfored, i.e. where the string already
            existed in `self.potential_strs`.
        nb_fuzzy : int
            Number of times where it was needed to match to nearest valid.
        self.fuzzy_map : dict
            Dictionary where the keys are the strings that were not valid and
            the values the strings they were matched to.

        """
        nb_to_match = len(strs_to_match)
        strs_matched = []
        nb_exact = 0
        nb_fuzzy = 0
        nb_ignore = 0
        start_time = time.time()

        for i, str_to_match in enumerate(strs_to_match):
            if (i + 1) % 1000 == 0:
                running_time = time.time() - start_time
                print(f'Progress: {round(i / nb_to_match * 100, 1)}%. ' +
                      f'Run time: {round(running_time, 1)} seconds. ' +
                      f'Per 1000 predictions: {round(running_time * 1000 / i, 1)} seconds. ' +
                      f'Exact: {nb_exact}. Fuzzy: {nb_fuzzy}. Ignored: {nb_ignore}. ' +
                      f'Number "cached": {len(self.fuzzy_map)}.'
                      )

            if str_to_match in self.ignore:
                nb_ignore += 1
                strs_matched.append(str_to_match)
            elif str_to_match in self.potential_strs:
                nb_exact += 1
                strs_matched.append(str_to_match)
            else:
                nb_fuzzy += 1

                if str_to_match in self.fuzzy_map.keys():
                    str_matched = self.fuzzy_map[str_to_match]
                else:
                    near_matches = difflib.get_close_matches(
                        str_to_match, self.potential_strs, n=1, cutoff=self.cutoff,
                        )
                    if len(near_matches) == 0:
                        str_matched = 'UNMATCHABLE'
                    else:
                        str_matched = near_matches[0]
                    self.fuzzy_map[str_to_match] = str_matched

                strs_matched.append(str_matched)

        strs_matched = np.array(strs_matched)

        return strs_matched, nb_exact, nb_fuzzy, nb_ignore


def _parse():
    parser = argparse.ArgumentParser(description='Summarize pred. files results.')

    # REQUIRED
    parser.add_argument(
        'file', type=str, default='',
        help='The file of predictions to match against a dictionary.',
        )
    parser.add_argument(
        '--dict', type=str, default='',
        help='Dictionary with valid outcomes to match up to.',
        )

    # OPTIONAL
    parser.add_argument(
        '--cutoff', type=float, default=0.0,
        help='The lower threshold to performin matching. Must be in [0, 1].',
        )
    parser.add_argument(
        '--ignore', type=str, nargs='+', default=['0=Mangler', 'bad cpd'],
        help='Cases to NOT match, such as e.g. empty, bad cpd, etc.'
        )

    args = parser.parse_args()

    return args


def _format_args(args) -> (pd.DataFrame, dict, float, set):
    print(f'Parsed args: {args}.')
    assert os.path.isfile(args.file)
    assert os.path.isfile(args.dict)
    assert 0 <= args.cutoff <= 1

    assert args.file[-4:].lower() == '.csv'
    assert args.dict[-4:].lower() == '.pkl'

    preds = pd.read_csv(args.file, na_values=[''], keep_default_na=False)
    dictionary = pickle.load(open(args.dict, 'rb'))

    assert isinstance(dictionary, set)
    assert 'pred' in preds.columns

    return preds, dictionary, args.cutoff, set(args.ignore)


def main():
    '''
    Perform matching of predictions against a dictionary. For instance, name
    predictions may be matched against a dictionary of valid names.

    '''
    args = _parse()
    preds, dictionary, cutoff, ignore = _format_args(args)

    matcher = MatchToStr(dictionary, cutoff, ignore)
    matched_strs, _, _, _ = matcher.match(preds['pred'].values)

    preds['pred'] = matched_strs

    print('Writing file!')
    path, fname = os.path.dirname(args.file), os.path.basename(args.file)
    fn_matched = ''.join((path, '/matched-', fname))

    if os.path.isfile(fn_matched):
        print(f'WARNING: File already exists: "{fn_matched}". Not writing!')
    else:
        print(f'Writing: "{fn_matched}."')
        preds.to_csv(fn_matched, index=False)


if __name__ == '__main__':
    main()
