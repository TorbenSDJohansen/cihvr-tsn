# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import string

import numpy as np

from timmsn.data.formatters import register_formatter


def _construct_maps():
    letters = sorted(string.ascii_lowercase) + ['æ', 'ø', 'å']
    assert len(letters) == len(set(letters))

    map_letter_idx = {letter: idx for idx, letter in enumerate(letters)}
    map_idx_letter = {v: k for k, v in map_letter_idx.items()}

    missing_indicator = max(map_idx_letter.keys()) + 1

    return map_letter_idx, map_idx_letter, missing_indicator


class NameFormatter():
    def __init__(
            self,
            label_format: str,
            max_name_len: int,
            min_name_len: int = 1,
            max_nb_names: int = None,
            min_nb_names: int = 1,
            name_separator: str = ' ', # TODO or allow list
            cast_to_empty: set = None, # TODO does this encompass bad cpd? No, not quite the same in CIHVR, i.e., not recorded weight vs. not transcribed on our side is not the same
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

        self.name_separator = name_separator
        self.cast_to_empty = cast_to_empty if cast_to_empty else set()

        self._assert_inputs()

        self.map_letter_idx, self.map_idx_letter, self.missing_indicator = _construct_maps()
        self.nb_chars_pr_element = len(self.map_idx_letter) + 1 # + 1 for missing indicator

        self.transform_label, self.clean_pred = self.map_format[label_format]

        self.num_classes = self.get_output_size()

    def _assert_inputs(self):
        assert self.label_format in self.map_format.keys()

        if self.label_format == 'last' or self.label_format == 'first':
            assert self.max_nb_names is None
            self.max_nb_names = 1
        else:
            assert isinstance(self.max_nb_names, int)

        assert self.max_name_len >= self.min_name_len >= 0
        assert self.max_nb_names >= self.min_nb_names >= 0

        assert isinstance(self.name_separator, str)
        assert isinstance(self.cast_to_empty, set)

    def get_output_size(self):
        '''
        Returns required output size.

        '''
        return [self.nb_chars_pr_element] * self.max_name_len * self.max_nb_names

    def _sanitize(self, raw_input: str) -> list:
        assert isinstance(raw_input, str)

        if raw_input in self.cast_to_empty:
            raw_input = ''

        split_input = raw_input.split(self.name_separator)

        return split_input

    def reorder_preds_individual_name(self, raw_pred: np.ndarray) -> np.ndarray:
        '''
        Reorders predictions to have all non-missings in front. Works on
        individual name, not to be called on multiple names such as "torben
        johansen". In such cases, all first on "torben", then on "johansen".

        '''
        assert len(raw_pred) == self.max_name_len

        non_missings = []

        for i, val in enumerate(raw_pred):
            if val != self.missing_indicator:
                non_missings.append(i)

        pred = np.concatenate([
            raw_pred[non_missings],
            np.ones(self.max_name_len - len(non_missings), dtype=int) * self.missing_indicator,
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
            label.append(self.map_letter_idx[char])

        label += (self.max_name_len - name_len) * [self.missing_indicator]

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
        pred = self.reorder_preds_individual_name(raw_pred)

        clean = []

        for idx in pred:
            if idx == self.missing_indicator:
                continue
            clean.append(self.map_idx_letter[idx])

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
        raw_input = self._sanitize(raw_input)

        assert len(raw_input) >= self.min_nb_names

        raw_input = raw_input[-1]
        label = self.transform_label_individual_name(raw_input)

        # Assert cycle consistency
        assert raw_input == self.clean_pred_individual_name(label)

        return label.astype('float')

    def transform_label_first_name(self, raw_input: str) -> np.ndarray:
        raw_input = self._sanitize(raw_input)

        assert len(raw_input) >= self.min_nb_names

        # TODO if len(raw_input) == 1, discard or use?
        raw_input = raw_input[0]
        label = self.transform_label_individual_name(raw_input)

        # Assert cycle consistency
        assert raw_input == self.clean_pred_individual_name(label)

        return label.astype('float')

    def transform_label_full_name(self, raw_input: str) -> np.ndarray:
        raise NotImplementedError

    def clean_pred_full_name(
            self, raw_pred: np.ndarray, assert_consistency: bool = True,
            ) -> str:
        raise NotImplementedError


@register_formatter
def last_name_long() -> NameFormatter:
    return NameFormatter('last', 18, 1)


@register_formatter
def first_name_long() -> NameFormatter:
    return NameFormatter('first', 18, 1)


@register_formatter
def last_name_long_cast_0() -> NameFormatter:
    return NameFormatter('last', 18, 0, cast_to_empty={'0=Mangler'})


@register_formatter
def first_name_long_cast_0() -> NameFormatter:
    return NameFormatter('first', 18, 0, cast_to_empty={'0=Mangler'})

# NOTES
# In '~/labels-root/210304-tab-b-cmd-tsdj-merge/nurse-name-{}.npy', using all
# three concatenated, the longest name is christophersen, at 14 letters. The
# second longest is at 12, such as christiansen. 3 cases of christophersen.
# The highest number of names is 5. However, only 1 case of 4 and 5, and only
# 19 of 1. This motivates numbers below, maybe make more tight.
# MAX_NAME_LEN = 14
# MAX_NB_NAMES = 5
