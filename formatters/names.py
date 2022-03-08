# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""
# pylint: disable=C0115, C0116


import string

import numpy as np

# from timmsn.data.formatters import register_formatter
register_formatter = lambda x: x


def _construct_maps():
    letters = sorted(string.ascii_lowercase) + ['æ', 'ø', 'å']
    assert len(letters) == len(set(letters))

    map_letter_idx = {letter: idx for idx, letter in enumerate(letters)}
    map_idx_letter = {v: k for k, v in map_letter_idx.items()}

    return map_letter_idx, map_idx_letter


MAP_LETTER_IDX, MAP_IDX_LETTER = _construct_maps()

ALLOWED_EMPTY = {'0=Mangler', ''} # perhaps dangerous to include 'empty' here
ALLOWED_BAD_CPD = {'bad cpd'} # dangerous to include 'b' here

MISSING_INDICATOR = max(MAP_IDX_LETTER.keys()) + 1
EMPTY_VALUE = '0=Mangler'

BAD_CPD_INDICATOR = MISSING_INDICATOR + 1
BAD_CPD_VALUE = 'bad cpd'


class NameFormatter():
    def __init__(
            self,
            label_format: str,
            max_name_len: int,
            min_name_len: int = 1,
            max_nb_names: int = None,
            min_nb_names: int = 1,
            name_separator: str = ' ',
            handle_bad_cpd: str = 'keep',
            handle_empty: str = 'keep',
        ):
        self.map_format = {
            'last': (self.transform_label_last_name, self.clean_pred_individual_name),
            'first': (self.transform_label_first_name, self.clean_pred_individual_name),
            'full': (self.transform_label_full_name, self.clean_pred_full_name),
            }
        self.label_format = label_format
        self.min_name_len = min_name_len
        self.max_name_len = max_name_len
        self.min_nb_names = min_nb_names
        self.max_nb_names = max_nb_names
        self.handle_bad_cpd = handle_bad_cpd
        self.handle_empty = handle_empty
        self.name_separator = name_separator

        self._asserts()
        self._instantiate_contants()
        self.transform_label, self.clean_pred = self.map_format[label_format]

    def _instantiate_contants(self):
        self.max_len = self.max_name_len * self.max_nb_names
        self.empty = np.array([MISSING_INDICATOR] * self.max_len)
        self.bad_cpd = np.array([BAD_CPD_INDICATOR] * self.max_len)
        self.num_classes = [len(MAP_IDX_LETTER) + 2] * self.max_len

    def _asserts(self):
        assert self.label_format in self.map_format.keys()

        if self.label_format == 'last' or self.label_format == 'first':
            assert self.max_nb_names is None
            self.max_nb_names = 1
        else:
            assert isinstance(self.max_nb_names, int)

        assert self.max_name_len >= self.min_name_len >= 0
        assert self.max_nb_names >= self.min_nb_names >= 0

        assert isinstance(self.name_separator, str)

    def _sanitize(self, raw_input: str) -> list:
        assert isinstance(raw_input, str)

        if raw_input in ALLOWED_EMPTY:
            return EMPTY_VALUE

        if raw_input in ALLOWED_BAD_CPD:
            return BAD_CPD_VALUE

        split_input = raw_input.split(self.name_separator)

        return split_input

    def reorder_preds_individual_name(self, raw_pred: np.ndarray) -> np.ndarray:
        '''
        Reorders predictions to have all non-missings in front. Works on
        individual name, not to be called on multiple names such as "torben
        johansen". In such cases, all first on "torben", then on "johansen".

        When some tolens are BAD_CPD_INDICATOR, these are cast to
        MISSING_INDICATOR in this method.
        '''
        assert len(raw_pred) == self.max_name_len

        non_missings = []

        for i, val in enumerate(raw_pred):
            if val < MISSING_INDICATOR:
                non_missings.append(i)

        pred = np.concatenate([
            raw_pred[non_missings],
            np.ones(self.max_name_len - len(non_missings), dtype=int) * MISSING_INDICATOR,
            ])

        return pred

    def transform_label_individual_name(self, raw_input: str) -> np.ndarray:
        '''
        Formats the name to array of floats representing characters. The floats
        are just integers cast to float, as that format is used for training
        the neural networks.

        '''
        name_len = len(raw_input)

        assert self.max_name_len >= name_len >= self.min_name_len

        label = []

        for char in raw_input:
            # Possible to have some option to map unkncown chars to unk-token
            label.append(MAP_LETTER_IDX[char])

        label += (self.max_name_len - name_len) * [MISSING_INDICATOR]

        label = np.array(label)

        # Assert cycle consistency
        assert raw_input == self.clean_pred_individual_name(label, False)

        return label.astype('float')

    def clean_pred_individual_name(
            self, raw_pred: np.ndarray, assert_consistency: bool = True,
        ) -> str:
        '''
        Maps predictions back from integer to string representation.

        '''
        nb_missing = sum(raw_pred == MISSING_INDICATOR)
        nb_bad_cpd = sum(raw_pred == BAD_CPD_INDICATOR)

        if (nb_missing + nb_bad_cpd) == self.max_len: # all missing or bad cpd
            if nb_bad_cpd > 0: # if at least one token bad cpd, cast to bad cpd
                return BAD_CPD_VALUE
            return EMPTY_VALUE # otherwise all are empty -> cast empty

        # Note: All BAD_CPD_INDICATOR cast to MISSING_INDICATOR here
        pred = self.reorder_preds_individual_name(raw_pred)

        clean = []

        for idx in pred:
            if idx >= MISSING_INDICATOR:
                continue
            clean.append(MAP_IDX_LETTER[idx])

        clean = ''.join((clean))

        # Need to be cycle consistent - however, the function may be called
        # from `transform_label*`, and we do not want infinite recursion, hence
        # the if.
        if assert_consistency:
            transformed_clean = self.transform_label_individual_name(clean)

            if not all(pred.astype('float') == transformed_clean):
                raise Exception(raw_pred, pred, clean, transformed_clean)

        return clean

    def transform_label_last_name(self, raw_input: str) -> np.ndarray:
        mod_input = self._sanitize(raw_input)

        if mod_input == EMPTY_VALUE:
            if self.handle_empty == 'keep':
                return self.empty
            if self.handle_empty == 'drop':
                return None

        if mod_input == BAD_CPD_VALUE:
            if self.handle_bad_cpd == 'keep':
                return self.bad_cpd
            if self.handle_bad_cpd == 'drop':
                return None

        assert len(mod_input) >= self.min_nb_names

        mod_input = mod_input[-1]
        label = self.transform_label_individual_name(mod_input)

        # Assert cycle consistency
        assert mod_input == self.clean_pred_individual_name(label)

        return label.astype('float')

    def transform_label_first_name(self, raw_input: str) -> np.ndarray:
        mod_input = self._sanitize(raw_input)

        if mod_input == EMPTY_VALUE:
            if self.handle_empty == 'keep':
                return self.empty
            if self.handle_empty == 'drop':
                return None

        if mod_input == BAD_CPD_VALUE:
            if self.handle_bad_cpd == 'keep':
                return self.bad_cpd
            if self.handle_bad_cpd == 'drop':
                return None

        assert len(mod_input) >= self.min_nb_names

        # TODO if len(raw_input) == 1, discard or use?
        raw_input = raw_input[0]
        label = self.transform_label_individual_name(mod_input)

        # Assert cycle consistency
        assert mod_input == self.clean_pred_individual_name(label)

        return label.astype('float')

    def transform_label_full_name(self, raw_input: str) -> np.ndarray:
        raise NotImplementedError

    def clean_pred_full_name(
            self, raw_pred: np.ndarray, assert_consistency: bool = True,
        ) -> str:
        raise NotImplementedError


@register_formatter
def last_name_keep_bad_cpd() -> NameFormatter:
    return NameFormatter('last', 18, 1)


@register_formatter
def last_name_drop_bad_cpd() -> NameFormatter:
    return NameFormatter('last', 18, 1, handle_bad_cpd='drop')


@register_formatter
def first_name_keep_bad_cpd() -> NameFormatter:
    return NameFormatter('first', 18, 1)


@register_formatter
def first_name_drop_bad_cpd() -> NameFormatter:
    return NameFormatter('first', 18, 1, handle_bad_cpd='drop')

# NOTES
# In '~/labels-root/210304-tab-b-cmd-tsdj-merge/nurse-name-{}.npy', using all
# three concatenated, the longest name is christophersen, at 14 letters. The
# second longest is at 12, such as christiansen. 3 cases of christophersen.
# The highest number of names is 5. However, only 1 case of 4 and 5, and only
# 19 of 1. This motivates numbers below, maybe make more tight.
# MAX_NAME_LEN = 14
# MAX_NB_NAMES = 5
