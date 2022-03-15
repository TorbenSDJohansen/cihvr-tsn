# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""
# pylint: disable=C0115, C0116


import numpy as np

from timmsn.data.formatters import register_formatter


ALLOWED_BAD_CPD = {'bad cpd', 'b'}

MISSING_INDICATOR = 0
EMPTY_TOKEN = ','
EMPTY_VALUE = ',:,'

BAD_CPD_INDICATOR = 1
BAD_CPD_VALUE = 'bad cpd'

MAP_NUM = {str(i): i + 2 for i in range(10)}
MAP_NUM.update({EMPTY_TOKEN: MISSING_INDICATOR})
MAP_NUM_INV = {v: k for k, v in MAP_NUM.items()}

MAP_MONTH = {str(i): i + 1 for i in range(1, 13)}
MAP_MONTH.update({EMPTY_TOKEN: MISSING_INDICATOR})
MAP_MONTH_INV = {v: k for k, v in MAP_MONTH.items()}

def _sanitize(raw_input: str) -> str:
    assert isinstance(raw_input, str), raw_input

    if raw_input in ALLOWED_BAD_CPD:
        return BAD_CPD_VALUE

    return raw_input.split(':')


def _strip_zero(elem: str):
    if elem.startswith(EMPTY_TOKEN):
        if len(elem) == 2 and elem[-1] != EMPTY_TOKEN:
            elem = elem[-1]
        else:
            return elem

    if len(elem) == 2 and elem.startswith('0'):
        elem = elem[-1] # lstrip first 0

    return elem


class DateFormatter:
    def __init__(
            self,
            handle_bad_cpd: str = 'keep',
            handle_empty: str = 'keep',
        ):
        self.max_len = 3
        self.handle_bad_cpd = handle_bad_cpd
        self.handle_empty = handle_empty

        self._asserts()
        self._instantiate_contants()

    def _instantiate_contants(self):
        self.empty = np.array([MISSING_INDICATOR] * self.max_len).astype(float)
        self.bad_cpd = np.array([BAD_CPD_INDICATOR] * self.max_len).astype(float)
        self.num_classes = [len(MAP_NUM) + 1] * 2 + [len(MAP_MONTH) + 1]
        # TODO limit first to smaller range (1 + 1 + 3 instead og 12)?

    def _asserts(self):
        assert self.handle_bad_cpd in ('keep', 'drop'), self.handle_bad_cpd
        assert self.handle_empty in ('keep', 'drop'), self.handle_empty

    def transform_label(self, raw_input: str) -> np.ndarray:
        mod_input = _sanitize(raw_input)

        if mod_input == BAD_CPD_VALUE:
            if self.handle_bad_cpd == 'keep':
                return self.bad_cpd
            if self.handle_bad_cpd == 'drop':
                return None

        if not len(mod_input) == 3: # if not day, month, and year avail. drop
            return None

        day, month, _ = mod_input
        day = _strip_zero(day)
        month = _strip_zero(month)

        if len(day) > 2: # sometimes label is invalid, e.g., 3 digit day
            return None

        if len(day) == 1:
            day_long = EMPTY_TOKEN + day
        else:
            day_long = day

        label = []

        for token in day_long:
            label.append(MAP_NUM.get(token, None)) # FIXME this allows days such as 41, 52, i.e. all [01-99]..

        label.append(MAP_MONTH.get(month, None))

        if None in label: # then one token of day or month is not valid -> drop
            return None

        label = np.array(label)

        # Assert consistency
        transformed_label = ':'.join([_strip_zero(x) for x in self.clean_pred(label, False, False).split(':')[:2]])
        comparison = ':'.join((day.replace(',,', ','), month)) # if day == ',,': day = ','
        assert comparison == transformed_label, (raw_input, comparison, transformed_label)

        return label.astype(float)

    def clean_pred(
            self,
            raw_pred: np.ndarray,
            assert_consistency: bool = True,
            strip_year: bool = True,
            ) -> str:
        nb_missing = sum(raw_pred == MISSING_INDICATOR)
        nb_bad_cpd = sum(raw_pred == BAD_CPD_INDICATOR)

        if (nb_missing + nb_bad_cpd) == self.max_len: # all missing or bad cpd
            if nb_bad_cpd > 0: # if at least one token bad cpd, cast to bad cpd
                return BAD_CPD_VALUE
            return EMPTY_VALUE # otherwise all are empty -> cast empty

        if nb_bad_cpd > 0:
            raw_pred[raw_pred == BAD_CPD_INDICATOR] = MISSING_INDICATOR

        day = MAP_NUM_INV[raw_pred[0]] + MAP_NUM_INV[raw_pred[1]]

        month = MAP_MONTH_INV[raw_pred[2]]

        if day == EMPTY_TOKEN + EMPTY_TOKEN:
            day = EMPTY_TOKEN
        elif day.startswith(EMPTY_TOKEN):
            day = day[-1]

        if len(day) == 1 and day != EMPTY_TOKEN:
            day = '0' + day

        if len(month) == 1 and month != EMPTY_TOKEN:
            month = '0' + month

        clean = ':'.join((day, month, 'YEAR-NOT-PREDICTED'))

        # Need to be cycle consistent - however, the function may be called from
        # `transform_label`, and we do not want infinite recursion, hence the if.
        if assert_consistency:
            transformed_clean = self.transform_label(clean)

            mod_pred = raw_pred.copy()
            if mod_pred[0] == 2: # in case pred first digit is 0, compare for when missing
                mod_pred[0] = 0

            if not (transformed_clean is None or all(mod_pred.astype('float') == transformed_clean)):
                raise Exception(raw_pred, clean, transformed_clean)

        if strip_year:
            clean = ':'.join(clean.split(':')[:2])

        return clean


class OldDateFormatter:
    def __init__(self):
        self.num_classes = [2, 4, 11, 13]

    def transform_label(self, raw_input: str) -> np.ndarray: # pylint: disable=R0911, R0912
        if not isinstance(raw_input, str):
            raise Exception(raw_input)

        if raw_input == 'bad cpd':
            return np.array([0, 0, 10, 0]).astype('float')

        split_input = raw_input.split(':')

        if len(split_input) != 3:
            return None

        day, month, year = split_input

        label = [1]

        if day == ',':
            label.extend([0, 10]) # 0 is empty 1st digit, 10 is "wildcard" 2nd digit
        elif len(day) == 2:
            for char in day:
                if char not in [str(x) for x in range(10)]:
                    return None
            if not (int(day[0]) in range(4) and int(day[1]) in range(10)):
                return None
            label.extend([int(day[0]), int(day[1])])
        elif len(day) == 1:
            if int(day) not in range(10):
                raise Exception(raw_input, day, month, year)
            label.extend([0, int(day)])
            day = ''.join(('0', day))
        else:
            return None

        if month == ',':
            label.append(0) # 0 for missing month
        else:
            if not (int(month) in range(1, 13) and len(month) in (1, 2)):
                return None # to drop
            label.append(int(month))

            if len(month) == 1:
                month = ''.join(('0', month))

        label = np.array(label)

        # Assert consistency.
        assert ':'.join((day, month)) == ':'.join((self.clean_pred(label, False, False).split(':')[:2])) # pylint: disable=C0301

        label = label.astype('float')

        return label

    def clean_pred(
            self,
            raw_pred: np.ndarray,
            assert_consistency: bool = True,
            strip_year: bool = True,
            ) -> str:
        if raw_pred[0] == 0:
            return 'bad cpd'

        pred = raw_pred.copy()

        if raw_pred[2] == 10:
            pred[1] = 0

        raw_day = pred[1:3]
        raw_month = pred[3]

        if raw_day[1] == 10:
            clean_day = ','
        else:
            clean_day = ''.join((str(raw_day[0]), str(raw_day[1])))

        if raw_month == 0:
            clean_month = ','
        else:
            if len(str(raw_month)) == 1:
                clean_month = ''.join(('0', str(raw_month)))
            else:
                clean_month = str(raw_month)

        clean = ':'.join((clean_day, clean_month, 'YEAR-NOT-PREDICTED'))

        if assert_consistency:
            transformed_clean = self.transform_label(clean)

            if not (transformed_clean is None or all(pred.astype('float') == transformed_clean)):
                raise Exception(raw_pred, pred, clean, transformed_clean)

        if strip_year:
            clean = ':'.join(clean.split(':')[:2])

        return clean

# TODO want to train on DARE with below formatter -> possible to CHANGE CIHVR to new, then no issues in leakage between train and test!!

@register_formatter
def dates_keep_bad_cpd() -> DateFormatter:
    return DateFormatter(handle_bad_cpd='keep')


@register_formatter
def dates_drop_bad_cpd() -> DateFormatter:
    return DateFormatter(handle_bad_cpd='drop')


@register_formatter
def dates_keep_bad_cpd_old() -> OldDateFormatter:
    return OldDateFormatter()


def _add_zero(elem: str):
    if elem == EMPTY_TOKEN:
        return elem

    if len(elem) == 1:
        elem = '0' + elem

    return elem


def test():
    f_new = dates_keep_bad_cpd()
    f_old = dates_keep_bad_cpd_old()

    dates = ['bad cpd']

    for day in [','] + [str(x) for x in range(1, 32)]:
        for month in [','] + [str(x) for x in range(1, 13)]:
            day = str(day)
            month = str(month)

            if len(day) == 1 and day != ',':
                sday = '0' + day
            else:
                sday = day
            if len(month) == 1 and month != ',':
                smonth = '0' + month
            else:
                smonth = month

            date = ':'.join((day, month, '93'))
            sdate = ':'.join((sday, smonth, '93'))

            dates.append(date)
            if date != sdate:
                dates.append(sdate)

    for date in dates:
        out_new = f_new.transform_label(date)
        out_old = f_old.transform_label(date)

        clean_new = f_new.clean_pred(out_new.astype(int))
        clean_old = f_old.clean_pred(out_old.astype(int))

        assert clean_new == clean_old, (date, clean_new, clean_old)

        if date != 'bad cpd':
            date_mod = ':'.join([_add_zero(x) for x in date.split(':')[:2]])
        else:
            date_mod = date

        assert clean_new == date_mod

    import itertools # pylint: disable=C0415

    for out in itertools.product(*[list(range(x)) for x in f_new.num_classes]):
        f_new.clean_pred(np.array(out))

    for out in itertools.product(*[list(range(x)) for x in f_old.num_classes]):
        f_old.clean_pred(np.array(out))


if __name__ == '__main__':
    test()
