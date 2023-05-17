# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import pandas as pd


def main():
    dst = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\data_to_dst\230510_upload.csv')
    distr = pd.read_csv(r'Y:\RegionH\Scripts\users\tsdj\data_to_dst\230510_upload_nurse_districts.csv')

    # Check for nurse name 1
    dst = dst.rename(columns={'nn_fn_m_1_pred': 'nn_fn', 'nn_ln_m_1_pred': 'nn_ln'})
    dst['nn'] = dst['nn_fn'] + ' ' + dst['nn_ln']
    distr['nn'] = distr['nn_fn'] + ' ' + distr['nn_ln']

    # Subset data
    dst = dst.drop_duplicates('Id')
    dst = dst[['nn', 'nn_fn', 'nn_ln']]
    distr = distr[['nn', 'nn_fn', 'nn_ln', 'rank']]
    distr_sub = distr[distr['rank'] == 1].copy()

    # Merge
    merge = dst.merge(distr_sub, on='nn', how='left')
    merge = dst.merge(distr_sub, on='nn_fn', how='left')
    merge = dst.merge(distr_sub, on='nn_ln', how='left')

    v = [x for x in dst['nn'].unique() if x in set(distr_sub['nn'])]
    v = [x for x in dst['nn_fn'].unique() if x in set(distr_sub['nn_fn'])]
    v = [x for x in dst['nn_ln'].unique() if x in set(distr_sub['nn_ln'])]


if __name__ == '__main__':
    main()
