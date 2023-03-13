# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

S2S style formatters

"""


import string

from typing import Tuple, Dict, Union

import numpy as np

from timmsn.data.formatters import (
    CharSeqFormatter,
    CharSeqSepSubsetFormatter,
    construct_map,
    register_formatter,
    )


def construct_numseq_keep_bad_cpd_map(bad_cpd_char: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = [str(x) for x in range(10)]
    chars.append(bad_cpd_char)

    return construct_map(chars)


def construct_names_keep_bad_cpd_map(bad_cpd_char: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(string.ascii_lowercase) + ['æ', 'ø', 'å']
    chars.append(bad_cpd_char)

    return construct_map(chars)


class CIHVRCharSeqFormatter(CharSeqFormatter):
    def __init__(self, handle_bad_cpd: str, bad_cpd_char: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(handle_bad_cpd, str):
            raise TypeError(f'handle_bad_cpd must be str, got {handle_bad_cpd}')

        if not handle_bad_cpd in ('keep', 'drop'):
            raise ValueError(f'handle_bad_cpd must be one of keep or drop, received {handle_bad_cpd}')

        if not isinstance(bad_cpd_char, str):
            raise TypeError(f'bad_cpd_char must be str, got {bad_cpd_char}')

        if len(bad_cpd_char) != 1:
            raise ValueError(f'bad_cpd_char must be 1 char, got {bad_cpd_char}')

        self.handle_bad_cpd = handle_bad_cpd
        self.bad_cpd_char = bad_cpd_char

    def sanitize(self, raw_input: Union[None, str]) -> Union[None, str]:
        if raw_input is None:
            return None

        if isinstance(raw_input, float): # if label loaded as float instead of str
            raw_input = int(raw_input)

        if isinstance(raw_input, int): # if label loaded as int instead of str
            raw_input = str(raw_input)

        if not isinstance(raw_input, str):
            raise TypeError(
                'Raw input must be string, None, float, or int, received input ' +
                f'{raw_input} of type {type(raw_input)}.'
                )

        if raw_input in self.allowed_empty:
            raw_input = '' # all '0=Mangler' cast to '' now

        if '=' in raw_input:
            raw_input = raw_input.split('=')[0]

        if '.' in raw_input:
            raw_input = str(int(float(raw_input))) # e.g., '3.0' -> '3'

        if raw_input == 'bad cpd':
            if self.handle_bad_cpd == 'drop':
                return None

            raw_input = self.bad_cpd_char

        return raw_input

    def clean_pred(self, raw_pred: np.ndarray, assert_consistency: bool = True) -> Union[str, None]: # input: array of non-negative ints
        clean = self.clean_seq(raw_pred, assert_consistency=assert_consistency)

        if clean == self.bad_cpd_char:
            clean = 'bad cpd'

        if clean == '':
            clean = self.empty_value # cast back to, e.g., '0=Mangler'

        return clean


class CIHVRNamesFormatter(CharSeqSepSubsetFormatter):
    def __init__(self, handle_bad_cpd: str, bad_cpd_char: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(handle_bad_cpd, str):
            raise TypeError(f'handle_bad_cpd must be str, got {handle_bad_cpd}')

        if not handle_bad_cpd in ('keep', 'drop'):
            raise ValueError(f'handle_bad_cpd must be one of keep or drop, received {handle_bad_cpd}')

        if not isinstance(bad_cpd_char, str):
            raise TypeError(f'bad_cpd_char must be str, got {bad_cpd_char}')

        if len(bad_cpd_char) != 1:
            raise ValueError(f'bad_cpd_char must be 1 char, got {bad_cpd_char}')

        self.handle_bad_cpd = handle_bad_cpd
        self.bad_cpd_char = bad_cpd_char

    def sanitize(self, raw_input: Union[None, str]) -> Union[None, str]:
        if raw_input is None:
            return None

        if not isinstance(raw_input, str):
            raise TypeError(
                'Raw input must be string, received input ' +
                f'{raw_input} of type {type(raw_input)}.'
                )

        if raw_input in self.allowed_empty:
            raw_input = '' # all '0=Mangler' cast to '' now

        if raw_input == 'bad cpd':
            if self.handle_bad_cpd == 'drop':
                return None

            raw_input = self.bad_cpd_char

        seq = self.cast_seps(raw_input).strip(self.sep_value)
        seq = self.grab_subseq(seq)

        return seq

    def clean_pred(self, raw_pred: np.ndarray, assert_consistency: bool = True) -> Union[str, None]: # input: array of non-negative ints
        clean = self.clean_seq(raw_pred, assert_consistency=assert_consistency)

        if clean == self.bad_cpd_char:
            clean = 'bad cpd'

        if clean == '':
            clean = self.empty_value # cast back to, e.g., '0=Mangler'

        return clean


class CIHVRDatesFormatter(CharSeqSepSubsetFormatter):
    def __init__(self, handle_bad_cpd: str, bad_cpd_char: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(handle_bad_cpd, str):
            raise TypeError(f'handle_bad_cpd must be str, got {handle_bad_cpd}')

        if not handle_bad_cpd in ('keep', 'drop'):
            raise ValueError(f'handle_bad_cpd must be one of keep or drop, received {handle_bad_cpd}')

        if not isinstance(bad_cpd_char, str):
            raise TypeError(f'bad_cpd_char must be str, got {bad_cpd_char}')

        if len(bad_cpd_char) != 1:
            raise ValueError(f'bad_cpd_char must be 1 char, got {bad_cpd_char}')

        self.handle_bad_cpd = handle_bad_cpd
        self.bad_cpd_char = bad_cpd_char

    def sanitize(self, raw_input: Union[None, str]) -> Union[None, str]:
        if raw_input is None:
            return None

        if not isinstance(raw_input, str):
            raise TypeError(
                'Raw input must be string, received input ' +
                f'{raw_input} of type {type(raw_input)}.'
                )

        if raw_input == 'bad cpd':
            if self.handle_bad_cpd == 'drop':
                return None

            raw_input = self.bad_cpd_char

        seq = self.cast_seps(raw_input).strip(self.sep_value)
        seq = self.grab_subseq(seq)

        for allowed_empty in self.allowed_empty:
            seq = seq.replace(allowed_empty, '')

        split = seq.split(self.sep_value)
        mod_seq = []

        for item in split: # remove leading 0s
            mod_seq.append(item.lstrip('0'))

        mod_seq = self.sep_value.join(mod_seq)

        return mod_seq

    def clean_pred(self, raw_pred: np.ndarray, assert_consistency: bool = True) -> Union[str, None]: # input: array of non-negative ints
        clean = self.clean_seq(raw_pred, assert_consistency=assert_consistency)

        if clean == self.bad_cpd_char: # FIXME what if just ANY part of clean is self.bad_cpd_char? Currently does not handle
            clean = 'bad cpd'

        split = clean.split(self.sep_value)
        mod_clean = []

        for item in split: # replace '' with proper ',' for missing
            mod_clean.append(item if item else self.empty_value)

        mod_clean = self.sep_value.join(mod_clean)

        return mod_clean


@register_formatter
def s2s_two_digit_keep_bad_cpd() -> CIHVRCharSeqFormatter:
    bad_cpd_char = 'b'
    map_char_idx, map_idx_char = construct_numseq_keep_bad_cpd_map(bad_cpd_char)

    formatter = CIHVRCharSeqFormatter(
        handle_bad_cpd='keep',
        bad_cpd_char=bad_cpd_char,
        max_seq_len=2, # this is before BOS and EOS tokens added
        map_char_idx=map_char_idx,
        map_idx_char=map_idx_char,
        allowed_empty={'', '0=Mangler'},
        empty_value='0=Mangler',
        unk_value='?',
        invalid_pred_value='InvalidPred',
        )

    return formatter


@register_formatter
def s2s_five_digit_keep_bad_cpd() -> CIHVRCharSeqFormatter:
    bad_cpd_char = 'b'
    map_char_idx, map_idx_char = construct_numseq_keep_bad_cpd_map(bad_cpd_char)

    formatter = CIHVRCharSeqFormatter(
        handle_bad_cpd='keep',
        bad_cpd_char=bad_cpd_char,
        max_seq_len=5, # this is before BOS and EOS tokens added
        map_char_idx=map_char_idx,
        map_idx_char=map_idx_char,
        allowed_empty={'', '0=Mangler'},
        empty_value='0=Mangler',
        unk_value='?',
        invalid_pred_value='InvalidPred',
        )

    return formatter


@register_formatter
def s2s_last_name_keep_bad_cpd() -> CIHVRNamesFormatter:
    bad_cpd_char = '*' # important part is **NOT** part of a-å
    map_char_idx, map_idx_char = construct_names_keep_bad_cpd_map(bad_cpd_char)

    formatter = CIHVRNamesFormatter(
        start=-1,
        end=None,
        allowed_seps={' '},
        sep_value=' ',
        handle_bad_cpd='keep',
        bad_cpd_char=bad_cpd_char,
        max_seq_len=18, # this is before BOS and EOS tokens added
        map_char_idx=map_char_idx,
        map_idx_char=map_idx_char,
        allowed_empty={'', '0=Mangler'},
        empty_value='0=Mangler',
        unk_value='?',
        invalid_pred_value='InvalidPred',
        )

    return formatter


@register_formatter
def s2s_first_name_keep_bad_cpd() -> CIHVRNamesFormatter:
    bad_cpd_char = '*' # important part is **NOT** part of a-å
    map_char_idx, map_idx_char = construct_names_keep_bad_cpd_map(bad_cpd_char)

    formatter = CIHVRNamesFormatter(
        start=0,
        end=1,
        allowed_seps={' '},
        sep_value=' ',
        handle_bad_cpd='keep',
        bad_cpd_char=bad_cpd_char,
        max_seq_len=18, # this is before BOS and EOS tokens added
        map_char_idx=map_char_idx,
        map_idx_char=map_idx_char,
        allowed_empty={'', '0=Mangler'},
        empty_value='0=Mangler',
        unk_value='?',
        invalid_pred_value='InvalidPred',
        )

    return formatter


@register_formatter
def s2s_dates_keep_bad_cpd() -> CIHVRDatesFormatter:
    bad_cpd_char = 'b'
    map_char_idx, map_idx_char = construct_numseq_keep_bad_cpd_map(bad_cpd_char)

    formatter = CIHVRDatesFormatter(
        start=0,
        end=2,
        allowed_seps={':'},
        sep_value=':',
        handle_bad_cpd='keep',
        bad_cpd_char=bad_cpd_char,
        max_seq_len=5, # this is before BOS and EOS tokens added
        map_char_idx=map_char_idx,
        map_idx_char=map_idx_char,
        allowed_empty={',', '0=Mangler'},
        empty_value=',',
        unk_value='?',
        invalid_pred_value='InvalidPred',
        )

    return formatter
