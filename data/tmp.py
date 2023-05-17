# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:44:16 2023

@author: sa-tsdj
"""



def nurse_merge_stats(wide: pd.DataFrame):
    wide = wide.copy()
    wide['full-name'] = wide['nn_fn'] + ' ' + wide['nn_ln']
    wide['ini-and-last'] = wide['nn_fn_i'] + ' ' + wide['nn_ln']

    nb_unique_first_and_last = wide['full-name'].nunique()
    nb_unique_ini_and_last = wide['ini-and-last'].nunique()
    nb_unique_first = wide['nn_fn'].nunique()
    nb_unique_ini = wide['nn_fn_i'].nunique()
    nb_unique_last = wide['nn_ln'].nunique()

    print(f'{nb_unique_first_and_last=}')
    print(f'{nb_unique_ini_and_last=}')
    print(f'{nb_unique_first=}')
    print(f'{nb_unique_ini=}')
    print(f'{nb_unique_last=}')


def merge_info_nurse_district(data: pd.DataFrame) -> pd.DataFrame: # FIXME update once merge strat finalized
    district_info = pd.read_csv(FN_NURSE_DISTRICT)

    for i in (1, 2, 3):
        ln_col, fn_col = f'nn_ln_m_{i}_pred', f'nn_fn_m_{i}_pred'

        tmp = district_info.copy() # tmp object where we can rename, drop etc
        tmp = tmp.rename(columns={
            'nn_ln': ln_col,
            'nn_fn': fn_col,
            **{c: f'nn_1_{c}' for c in tmp.columns if c not in ('nn_ln', 'nn_fn', 'nn_fn_i')},
            })
        data = data.merge(tmp, on=[ln_col, fn_col], how='left')

        # Names to hash
        data[ln_col] = _names_to_numeric(data[ln_col].values, False)
        data[fn_col + '_ini'] = _names_to_numeric(data[fn_col].values, True) # pylint: disable=C0301
        data[fn_col] = _names_to_numeric(data[fn_col].values, False)

    # TODO
    # Also want to do non-exact matching... including using initials
    # But with fn initials + ln we have duplicated in district_info

    raise NotImplementedError

    return data





def make_hist(vals: pd.Series, fname: str = None, bins: int = 100): # FIXME (re)move
    from matplotlib import pyplot as plt

    plt.hist(vals, bins=bins)
    plt.xlabel('Number of Appearances in Journals')
    plt.ylabel('Counts')

    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def merge_info_nurse_district_stats(main: pd.DataFrame): # FIXME (re)move
    data = main.drop_duplicates('Filename').copy()
    data['full-nurse-name'] = data['nn_fn_m_1_pred'] + ' ' + data['nn_ln_m_1_pred']

    # Drop if both first and last name 0=Mangler
    data = data[data['full-nurse-name'] != '0=Mangler 0=Mangler']

    # Still some, but only few, where one of the two now missing
    print('Number 0=Mangler only for fn: {}'.format((data['nn_fn_m_1_pred'] == '0=Mangler').sum()))
    print('Number 0=Mangler only for ln: {}'.format((data['nn_ln_m_1_pred'] == '0=Mangler').sum()))

    # Number uniques
    print('Unique full names: {}'.format(data['full-nurse-name'].nunique()))
    print('Unique first names: {}'.format(data['nn_fn_m_1_pred'].nunique()))
    print('Unique last names: {}'.format(data['nn_ln_m_1_pred'].nunique()))

    # Histograms of number of appearances of nurses in journals
    out_dir = r'W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\on-nurse-district-match'
    name_cols = ['full-nurse-name', 'nn_fn_m_1_pred', 'nn_ln_m_1_pred']

    for col in name_cols:
        make_hist(
            vals=data[col].value_counts(),
            fname=os.path.join(out_dir, f'{col}.png'),
            )

    # Very skewed, see how many appear 1 time only
    print(data['full-nurse-name'].value_counts().value_counts())

    # Trim those away and make plots again
    counts = data['full-nurse-name'].value_counts().reset_index()
    counts.columns = ['full-nurse-name', 'counts']
    data = data.merge(counts, on='full-nurse-name', how='left')
    sub_1 = data[data['counts'] > 1]

    # New histograms
    for col in name_cols:
        make_hist(
            vals=sub_1[col].value_counts(),
            fname=os.path.join(out_dir, f'{col}-subset-1.png'),
            )

    # Stronger subset
    sub_5 = data[data['counts'] > 5]

    # New histograms
    for col in name_cols:
        make_hist(
            vals=sub_5[col].value_counts(),
            fname=os.path.join(out_dir, f'{col}-subset-5.png'),
            )

    # Stronger subset
    sub_100 = data[data['counts'] > 100]

    # New histograms
    for col in name_cols:
        make_hist(
            vals=sub_100[col].value_counts(),
            fname=os.path.join(out_dir, f'{col}-subset-100.png'),
            bins=20,
            )

    print(f'Number remaining after threshold > 100: {len(sub_100)} of {len(data)}')

    # Now checks on merge
    district_info = pd.read_csv(FN_NURSE_DISTRICT)
    district_info['full-nurse-name'] = district_info['nn_fn'] + ' ' +  district_info['nn_ln']
    district_info['merged'] = 1.0 # this will then be NaN for non-merged after left merge

    # No duplicate names here
    assert len(district_info) == len(set(district_info['full-nurse-name']))

    merge = data.merge(district_info, on='full-nurse-name', how='left')

    nb_matched = (~merge['merged'].isnull()).sum()
    print(f'Number matched: {nb_matched} of {len(merge)}')

    most_common_unmatched = merge.loc[merge['merged'].isnull(), 'full-nurse-name'].value_counts()
    print(most_common_unmatched)

    sub_merged = merge[merge['merged'] == 1.0]

    # New histograms
    for col in name_cols:
        make_hist(
            vals=sub_merged[col].value_counts(),
            fname=os.path.join(out_dir, f'{col}-subset-merged.png'),
            bins=20,
            )