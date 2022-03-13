# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""
# pylint: disable=C0115, C0116


import numpy as np

from timmsn.data.formatters import register_formatter


ALLOWED_EMPTY = {'0=Mangler', 'empty', ''}
ALLOWED_BAD_CPD = {'bad cpd', 'b'}

MISSING_INDICATOR = 10
EMPTY_VALUE = '0=Mangler'

BAD_CPD_INDICATOR = 11
BAD_CPD_VALUE = 'bad cpd'

def _sanitize(raw_input: str or int or float) -> str:
    assert isinstance(raw_input, (str, int, float)), raw_input

    if raw_input in ALLOWED_EMPTY:
        return EMPTY_VALUE

    if raw_input in ALLOWED_BAD_CPD:
        return BAD_CPD_VALUE

    if isinstance(raw_input, float):
        assert int(raw_input) == raw_input, raw_input

    if isinstance(raw_input, str) and '=' in raw_input:
        mod_input = str(int(raw_input.split('=')[0]))
    else:
        mod_input = str(int(raw_input))

    return mod_input


class NumSeqFormatter:
    def __init__(
            self,
            max_len: int,
            min_len: int = 1,
            handle_bad_cpd: str = 'keep',
            handle_empty: str = 'keep',
        ):
        self.max_len = max_len
        self.min_len = min_len
        self.handle_bad_cpd = handle_bad_cpd
        self.handle_empty = handle_empty

        self._asserts()
        self._instantiate_contants()

    def _instantiate_contants(self):
        self.empty = np.array([MISSING_INDICATOR] * self.max_len).astype(float)
        self.bad_cpd = np.array([BAD_CPD_INDICATOR] * self.max_len).astype(float)
        self.num_classes = [BAD_CPD_INDICATOR] * self.max_len

    def _asserts(self):
        assert isinstance(self.max_len, int), self.max_len
        assert isinstance(self.min_len, int), self.min_len
        assert self.max_len >= self.min_len >= 0, (self.max_len, self.min_len)
        assert self.handle_bad_cpd in ('keep', 'drop'), self.handle_bad_cpd
        assert self.handle_empty in ('keep', 'drop'), self.handle_empty

    def transform_label(self, raw_input: str or int or float) -> np.ndarray:
        mod_input = _sanitize(raw_input)

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

        input_len = len(mod_input)
        assert self.max_len >= input_len >= self.min_len, raw_input

        label = [MISSING_INDICATOR] * (self.max_len - input_len)

        for token in mod_input:
            label.append(int(token))

        label = np.array(label)

        # Assert consistency
        assert mod_input == self.clean_pred(label, False)

        return label.astype(float)

    def clean_pred(self, raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
        nb_missing = sum(raw_pred == MISSING_INDICATOR)
        nb_bad_cpd = sum(raw_pred == BAD_CPD_INDICATOR)

        if (nb_missing + nb_bad_cpd) == self.max_len: # all missing or bad cpd
            if nb_bad_cpd > 0: # if at least one token bad cpd, cast to bad cpd
                return BAD_CPD_VALUE
            return EMPTY_VALUE # otherwise all are empty -> cast empty

        # if here: at least one token neither missing nor bad cpd
        # assumption: in such cases, extract value better than cast based on
        # partial missing or bad cpd info
        # alternative: if partial bad cpd, could also cast to bad cpd

        clean = []

        for val in raw_pred: # input, left to right, all "real" tokens
            if val >= MISSING_INDICATOR:
                continue
            clean.append(val)

        len_clean_pred = len(clean)
        assert len_clean_pred > 0, (raw_pred, clean)

        # Place all empty fields first
        reordered_pred = np.array([MISSING_INDICATOR] * (self.max_len - len_clean_pred) + clean)
        clean = ''.join((str(x) for x in clean))

        # Need to be cycle consistent - however, the function may be called from
        # `transform_label`, and we do not want infinite recursion, hence the if.
        if assert_consistency:
            transformed_clean = self.transform_label(clean)

            if not (transformed_clean is None or all(reordered_pred.astype('float') == transformed_clean)):
                raise Exception(raw_pred, reordered_pred, clean, transformed_clean)

        return clean


@register_formatter
def weight_keep_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(5, 4, 'keep', 'keep')


@register_formatter
def weight_drop_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(5, 4, 'drop', 'keep')


@register_formatter
def lenght_keep_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(2, 2, 'keep', 'keep')


@register_formatter
def lenght_drop_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(2, 2, 'drop', 'keep')


@register_formatter
def one_digit_keep_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(1, 1, 'keep', 'keep')


@register_formatter
def one_digit_drop_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(1, 1, 'drop', 'keep')


@register_formatter
def two_digit_keep_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(2, 1, 'keep', 'keep')


@register_formatter
def two_digit_drop_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(3, 1, 'drop', 'keep')


@register_formatter
def three_digit_keep_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(3, 1, 'keep', 'keep')


@register_formatter
def three_digit_drop_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(3, 1, 'drop', 'keep')


@register_formatter
def four_digit_keep_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(4, 1, 'keep', 'keep')


@register_formatter
def four_digit_drop_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(4, 1, 'drop', 'keep')


@register_formatter
def five_digit_keep_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(5, 1, 'keep', 'keep')


@register_formatter
def five_digit_drop_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(5, 1, 'drop', 'keep')
