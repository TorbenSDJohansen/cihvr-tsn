# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

S2S style formatters

"""


from typing import Tuple, Dict, Union

from timmsn.data.formatters import (
    CharSeqFormatter,
    construct_map,
    register_formatter,
    )


def construct_numseq_map() -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = [str(x) for x in range(10)]

    return construct_map(chars)


def construct_numseq_keep_bad_cpd_map() -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = [str(x) for x in range(10)]
    chars.append('b') # for bad cpd

    return construct_map(chars)


class NumSeqFormatter(CharSeqFormatter):
    def __init__(self, handle_bad_cpd: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(handle_bad_cpd, str):
            raise TypeError(f'handle_bad_cpd must be str, go {handle_bad_cpd}')

        if not handle_bad_cpd in ('keep', 'drop'):
            raise ValueError(f'handle_bad_cpd must be one of keep or drop, received {handle_bad_cpd}')

        self.handle_bad_cpd = handle_bad_cpd

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

        if raw_input == 'bad cpd':
            if self.handle_bad_cpd == 'drop':
                return None

            raw_input = 'b'

        return raw_input


@register_formatter
def s2s_two_digit_keep_bad_cpd() -> NumSeqFormatter:
    map_char_idx, map_idx_char = construct_numseq_keep_bad_cpd_map()

    formatter = NumSeqFormatter(
        handle_bad_cpd='keep',
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
def s2s_five_digit_keep_bad_cpd() -> NumSeqFormatter:
    map_char_idx, map_idx_char = construct_numseq_keep_bad_cpd_map()

    formatter = NumSeqFormatter(
        handle_bad_cpd='keep',
        max_seq_len=5, # this is before BOS and EOS tokens added
        map_char_idx=map_char_idx,
        map_idx_char=map_idx_char,
        allowed_empty={'', '0=Mangler'},
        empty_value='0=Mangler',
        unk_value='?',
        invalid_pred_value='InvalidPred',
        )

    return formatter
