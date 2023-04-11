# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Extracted from data/gen_labels.py, collection of functions to verify the label
data is in correct format, such as, e.g., no weights of value 100, no "4" in a
Table B cell with valid values 1-3, etc.

"""


import string
import time

from typing import Union


class Verifiers: # pylint: disable=C0115
    _allowed_chars_names = set(list(string.ascii_lowercase) + ['æ', 'ø', 'å'])

    @staticmethod
    def verify_weight(weight): # pylint: disable=C0116
        if weight in ('bad cpd', 'empty'):
            return weight

        if float(weight) == int(float(weight)):
            weight = int(float(weight))

        _allowed_range = set(range(1000, 20_000)) # Maybe change?

        if int(weight) not in _allowed_range:
            print(f'Bad weight value: {weight}. Casting to None.')
            return None

        return weight

    @staticmethod
    def verify_length(length): # pylint: disable=C0116
        if length in ('bad cpd', '0=Mangler'):
            return length

        if float(length) == int(float(length)):
            length = int(float(length))

        _allowed_range = set(range(20, 100)) # Maybe change?

        if int(length) not in _allowed_range:
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

        tab_b_entry = {0.0: '0=Mangler', 1.0: '1=god', 2.0: 'middel', 3.0: 'dårlig'}.get(tab_b_entry, tab_b_entry)
        _allowed = {1, 2, 3}

        if int(tab_b_entry[0]) not in _allowed:
            print(f'Bad table B (1, 2, 3) value: {tab_b_entry}. Casting to None.')
            return None

        return tab_b_entry

    @staticmethod
    def verify_tab_b_12(tab_b_entry): # pylint: disable=C0116
        if tab_b_entry in ('bad cpd', '0=Mangler'):
            return tab_b_entry

        tab_b_entry = {0.0: '0=Mangler', 1.0: '1=ja', 2.0: 'nej'}.get(tab_b_entry, tab_b_entry)
        _allowed = {1, 2}

        if int(tab_b_entry[0]) not in _allowed:
            print(f'Bad table B (1, 2) value: {tab_b_entry}. Casting to None.')
            return None

        return tab_b_entry

    @staticmethod
    def verify_tab_b_int(tab_b_entry): # pylint: disable=C0116
        if tab_b_entry in ('bad cpd', '0=Mangler'):
            return tab_b_entry

        if float(tab_b_entry) == int(float(tab_b_entry)):
            tab_b_entry = int(float(tab_b_entry))

        _allowed = set(range(24))

        if int(tab_b_entry) not in _allowed:
            print(f'Bad table B (int) value: {tab_b_entry}. Casting to None.')
            return None

        return tab_b_entry

    @staticmethod
    def verify_bfdurany(duration): # pylint: disable=C0116
        if duration in ('bad cpd', '0=Mangler'):
            return duration

        if float(duration) == int(float(duration)):
            duration = int(float(duration))

        _allowed = set(range(14)) # from "tasteinstruktion"

        if int(duration) not in _allowed:
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

    def verify_preterm_birth_weeks(self, weeks: Union[str, int, float]) -> str:
        if weeks == 'bad cpd':
            return weeks

        if isinstance(weeks, float):
            if not int(weeks) == weeks:
                raise ValueError(f'if weeks is float, must have no decimal, but got {weeks}')
            weeks = int(weeks)
        if isinstance(weeks, int):
            weeks = str(weeks)
        elif isinstance(weeks, str):
            if float(weeks) == int(float(weeks)): # cast "0.0" to "0" etc
                weeks = str(int(float(weeks)))
        else:
            raise TypeError(f'weeks must be type str, int, or float, got {weeks} of type {type(weeks)}')

        if weeks == '0': # "mis-labelled" and should be 0=Mangler
            weeks = '0=Mangler'
        else: # should be str representing int value now
            weeks_as_int = int(weeks)
            if not 0 < weeks_as_int < 30:
                raise ValueError(f'number weeks preterm value {weeks_as_int} not realistic')

        return weeks
