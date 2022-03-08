# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import numpy as np

from timmsn.data.formatters import register_formatter


ALLOWED_EMPTY = {'0=Mangler', 'empty', ''}
ALLOWED_BAD_CPD = {'bad cpd', 'b'}

MISSING_INDICATOR = 10
EMPTY_VALUE = '0=Mangler'

BAD_CPD_INDICATOR = 11 # TODO even use this?
BAD_CPD_VALUE = 'bad cpd'

def _sanitize(raw_input: str or int or float) -> str:
    assert isinstance(raw_input, (str, int, float)), raw_input

    if raw_input in ALLOWED_EMPTY:
        return EMPTY_VALUE

    if raw_input in ALLOWED_BAD_CPD:
        return BAD_CPD_VALUE

    if isinstance(raw_input, float):
        assert int(raw_input) == raw_input

    mod_input = str(int(raw_input)) # strip zeros -> cast str

    return mod_input


class NumSeqFormatter:
    def __init__(
            self,
            max_len: int,
            min_len: int = 0,
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
        self.empty = np.array([MISSING_INDICATOR] * self.max_len)
        self.bad_cpd = np.array([BAD_CPD_INDICATOR] * self.max_len)
        self.num_classes = [BAD_CPD_INDICATOR] * self.max_len # FIXME if drop, e.g., bad cpd, can have fewer classes for each token

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
        assert self.max_len >= input_len >= self.min_len

        label = [MISSING_INDICATOR] * (self.max_len - input_len)

        for token in mod_input:
            label.append(int(token))

        label = np.array(label)

        # Assert consistency
        assert mod_input == self.clean_pred(label, False)

        return label.astype(float)

    def clean_pred(self, raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
        # if some-method-to-eval-empty:
            # return EMPTY_VALUE

        # if some-method-to-eval-bad cpd:
            # return BAD_CPD_VALUE

        
        clean = []
        reordered_pred = []

        for val in raw_pred: # input, left to right, all "real" tokens
            if val >= 10:
                continue
            reordered_pred.append(val)
            clean.append(str(val))

        
        reordered_pred = np.array([MISSING_INDICATOR] + reordered_pred)

        assert len(clean) > 0, (raw_pred, clean)

        clean = ''.join(clean)

        # Need to be cycle consistent - however, the function may be called from
        # `transform_label`, and we do not want infinite recursion, hence the if.
        if assert_consistency:
            transformed_clean = self.transform_label(clean)
            # assert transformed_clean is None or all(pred.astype('float') == transformed_clean)
    
            if not (transformed_clean is None or all(pred.astype('float') == transformed_clean[1:])):
                raise Exception(raw_pred, pred, clean, transformed_clean)






@register_formatter
def weight_keep_bad_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(5, 4, 'keep', 'keep')


@register_formatter
def weight_keep_drop_cpd() -> NumSeqFormatter:
    return NumSeqFormatter(5, 4, 'drop', 'keep')
