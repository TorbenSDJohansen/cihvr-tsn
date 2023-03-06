# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import os

import numpy as np


def main():
    # FIXME this needs to be updated to work again.
    root = 'Y:/RegionH/Scripts/users/tsdj/storage/labels-root/'
    nb_docs = 95_323

    root_train = ''.join((root, '210304-tab-b-cmd-tsdj-merge/'))
    root_test_1 = ''.join((root_train, 'test/'))
    root_test_2 = ''.join((root_train, 'tab_b_test_all_cells/'))

    fnames_train = [
        f'{root_train}{x}' for x in os.listdir(root_train)
        if os.path.isfile(f'{root_train}{x}')]
    fnames_test_1 = [
        f'{root_test_1}{x}' for x in os.listdir(root_test_1)
        if os.path.isfile(f'{root_test_1}{x}')]
    fnames_test_2 = [
        f'{root_test_2}{x}' for x in os.listdir(root_test_2)
        if os.path.isfile(f'{root_test_2}{x}')]
    fnames_test = fnames_test_1 + fnames_test_2

    exclude_cells = [
        '*-c5-*', '*-c6-*', '*-c9-*', '*-c10-*', '*-c11-*', '*-c12-*',
        '*-c13-*', '*-c14-*', '*-c16-*',
        ]
    for exclude in exclude_cells:
        if exclude[0] == '*' == exclude[-1]:
            exclude = exclude[1:-1]
            fnames_train = [x for x in fnames_train if exclude not in x.split('/')[-1]]
            fnames_test = [x for x in fnames_test if exclude not in x.split('/')[-1]]
        else:
            raise Exception('Only supports double wildcard exclusion as of now.')


    nb_cells = len(set(x.split('/')[-1] for x in fnames_test))

    labels_train = np.concatenate([np.load(x, allow_pickle=True) for x in fnames_train])[:, 1]
    labels_train = labels_train[labels_train != 'bad cpd']
    # labels_train = labels_train.astype(str)
    # v, c = np.unique(labels_train, return_counts=True)
    # import pandas as pd
    # d = pd.DataFrame({'v': v, 'c': c})

    labels_test = np.concatenate([np.load(x, allow_pickle=True) for x in fnames_test])[:, 1]
    labels_test = labels_test[labels_test != 'bad cpd']

    share_train = round(len(labels_train) / (nb_cells * nb_docs) * 100, 1)
    share_test = round(len(labels_test) / (nb_cells * nb_docs) * 100, 1)

    print(f'Share of total cells used for training + validation: {share_train}%.')
    print(f'Share of total cells used for testing: {share_test}%.')
    print(f'Total number of test cells: {len(labels_train)}.')
    print(f'Total number of test cells: {len(labels_test)}.')


if __name__ == '__main__':
    main()
