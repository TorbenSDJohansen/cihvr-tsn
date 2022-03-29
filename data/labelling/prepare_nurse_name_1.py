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


import os # os.path.split() to get name and folder

import json

import pandas as pd


def main():
    r'''
    Take as input predictions of first and last name. Could be matched, could
    also be raw.
    Remember to drop those already labelled.
    Select proper subset (see notes above).
    Prepare format for ens app, see json.load(open(r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-tab-b\Torben_tab_b.json', 'r'))
    Then label manually with these as initial predictions.
    Finally map to label format.
    '''
    ourdir = r'Y:\RegionH\Scripts\users\tsdj\storage\datasets\manual-nurse-name-1'


if __name__ == '__main__':
    main()
