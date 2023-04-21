# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Purpose of this script: Whenever uploads are made to DST, data is prepared here
(from now on). This is then updated as we go along with new uploads.

QUESTION: How to add a new dataset?
ANSWER: Add the filename as a global, write function_prepare_, add new info to
global DATASETS, and change function load_prepare_merge appropriately.

QUESTION: How is data merged?
ANSWER: This depends on source. For "external" sources, "Id" is used. However,
this includes 11 duplicates. For all data generated at SDU, we merge on
"Filename", which then means that duplicates will both get all the information,
solving any potential issues.
"""

# FIXME CHECK if new ID column(s) added, NEED to drop before DST upload
# TODO merge nurse district in some way... maybe merge wrt nurse name, maybe
# upload as separate file - BUT if they re-key nurse name, we need to merge on
# before upload!

import os
import csv
import pickle
import datetime
import string
import math

import numpy as np
import pandas as pd


LETTERS = list(string.ascii_lowercase) + ['æ', 'ø', 'å', ' ']
LETTERS_SET = set(LETTERS)

# GLOBALS

## EXTERNAL
FN_MAIN = 'Y:/RegionH/SPJ/Database/export_181106.txt'
FN_CPR = 'Y:/RegionH/SPJ/CPR info and duplicate info/spj_cprlist-plusID_180227_181106.dta'
FN_STATUS = 'Y:/RegionH/Scripts/users/tsdj/storage/datasets/cprstatus_match_final.dta'

## LONG TABLE
FN_CLUSTER = 'Y:/RegionH/Scripts/data/treatment_pages_classifications.pkl'
FN_INTENSITY = 'Y:/RegionH/Scripts/users/jfl/Treatment_intensity/intensity_df_unsup.pkl'
FN_INTENSITY_R2 = 'Y:/RegionH/Scripts/users/jfl/Treatment_intensity/intensity_df_sup_r2.pkl'

## CELLS (PREDICTIONS)
FN_BF7DO_PRED = r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\bf7do\circle-s2s\wide-preds.csv'
FN_DABF_PRED =  r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\dabf\int-s2s-5d-restrict-2d\wide-preds.csv'
FN_DATE_PRED = r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\date\mh\wide-preds.csv'
FN_LENGTH_PRED = r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\length\int-s2s-5d-restrict-2d\wide-preds.csv'
FN_NURSE_LASTNAME_PRED =  r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\s2s-tl\wide-preds.csv'
FN_NURSE_FIRSTNAME_PRED = r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\mh-tl\wide-preds.csv'
FN_PRETERM_PRED = r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm\circle-s2s\wide-preds.csv'
FN_PRETERM_WEEKS_PRED = r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm-wks\int-s2s-5d-restrict-2d\wide-preds.csv'
FN_WEIGHT_PRED = r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\weight\mh\wide-preds.csv'
FN_TAB_B_PRED =  r'Z:\faellesmappe\tsdj\cihvr-timmsn\pred\tab-b\int-s2s-5d-restrict-2d\wide-preds.csv'

def _prepare_main() -> pd.DataFrame:
    main = pd.read_csv(FN_MAIN, sep=';', na_values=[''], keep_default_na=False)
    bad_cols = [
        'FileHash', 'JournalId', # some ids
        'FirstName', 'LastName', # names
        'MotherFirstName', 'MotherLastName', 'MotherBirthName', # mother names
        'FartherFirstName', 'FartherLastName', # father names
        'pocc', 'mocc', # occupations
        'notes', # highly specific
        ]

    assert set(bad_cols).intersection(set(main.columns)) == set(bad_cols)

    main = main.drop(columns=bad_cols) # pylint: disable=E1101

    return main


def _prepare_cpr() -> pd.DataFrame:
    cpr = pd.read_stata(FN_CPR)

    cpr['dob'] = cpr['cpr'].apply(lambda x: x[:-4])
    cpr['bdd'] = cpr['dob'].apply(lambda x: x[:2])
    cpr['bdm'] = cpr['dob'].apply(lambda x: x[2:4])
    cpr['bdy'] = cpr['dob'].apply(lambda x: x[4:])

    cpr = cpr.rename(columns={'id_s': 'Id'})
    cpr = cpr.drop(columns=['id_c', 'dob'])

    return cpr


def _prepare_status() -> pd.DataFrame:
    status = pd.read_stata(FN_STATUS)

    status = status.rename(columns={'id': 'Id', 'sex': 'sex_revised'})

    return status


def _prepare_cluster() -> pd.DataFrame:
    with open(FN_CLUSTER, 'rb') as file:
        cluster = pickle.load(file)

    cluster['Filename'] = list(map(rename, cluster['path'].values))

    cluster['embedded_x'] = round(cluster['embedded_x'], 2)
    cluster['embedded_y'] = round(cluster['embedded_y'], 2)

    cluster = drop_not_imported_entries(cluster)
    cluster['page'] = create_page_number_col(cluster).values
    cluster = cluster.drop(columns={'path', 'embedded_x', 'embedded_y'})

    return cluster


def _prepare_intensity() -> pd.DataFrame:
    # NOTE: Row number wrong ordering fixed at DST
    with open(FN_INTENSITY, 'rb') as file:
        intensity_r1 = pickle.load(file)

    with open(FN_INTENSITY_R2, 'rb') as file:
        intensity_r2 = pickle.load(file)

    intensity = pd.concat([intensity_r1, intensity_r2]).reset_index()
    intensity['Filename'] = list(map(rename, intensity['path'].values))

    to_recode = ['empty2', 'empty1', '36mon', '30mon', '24mon', '18mon', '15mon']

    for col in to_recode:
        is_array = intensity[col].apply(lambda x: isinstance(x, np.ndarray))
        intensity.loc[is_array, col] = list(map(
            recode_array_str,
            intensity[is_array][col].values,
            ))

    intensity = drop_not_imported_entries(intensity)
    intensity['page'] = create_page_number_col(intensity).values
    intensity['row'] = create_row_number(intensity).values
    intensity = intensity.drop(columns=['path', 'index', 'empty1', 'empty2'])

    return intensity


def _prepare_weight() -> pd.DataFrame:
    weight = pd.read_csv(FN_WEIGHT_PRED, na_values=[''], keep_default_na=False)
    weights_sequence = [
        'bweight', 'weightv2', 'weightv3', 'weightv4',
        'weightv5', 'weightv6', 'weightv7', 'weightv8',
        ]

    weight = weight.rename(columns={'journal': 'Filename'})
    weight = round_prob(weight, 2)

    flag_weight = verify_sequence_nondecreasing(
        weight, [''.join((x, '_pred')) for x in weights_sequence]
        )
    weight['incons_weight_seq'] = flag_weight == 0

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    weight = weight.replace('bad cpd', np.nan)

    return weight


def _prepare_date() -> pd.DataFrame:
    date = pd.read_csv(FN_DATE_PRED, na_values=[''], keep_default_na=False)
    main = _prepare_main()
    dates_sequence = [
        'datev2', 'datev3', 'datev4', 'datev5', 'datev6', 'datev7', 'datev8',
        ]

    bdate = '19' + (
        main['BirthDateYear'].astype(str).str.split('.').apply(lambda x: x[0]) +
        ':' + main['BirthDateMonth'].astype(str).str.split('.').apply(lambda x: x[0]) +
        ':' + main['BirthDateDay'].astype(str).str.split('.').apply(lambda x: x[0])
        )
    main['bdate'] = pd.to_datetime(bdate, format='%Y:%m:%d', errors='coerce')
    main = main[['Filename', 'bdate']]

    date = date.rename(columns={'journal': 'Filename'})
    date = round_prob(date, 2)

    # If missing bdate, replace with non-missing value. Then drop duplicates.
    duplicated = main.duplicated('Filename', keep=False)
    main.loc[duplicated, 'bdate'] = main[duplicated].groupby('Filename')['bdate'].apply(
        lambda x: x.replace(np.nan, x.dropna().iloc[0])
        )
    main = main.drop_duplicates('Filename')

    # We can now merge bdate onto the date predictions.
    date = date.merge(main, on='Filename', how='left')

    # Modify predictions by adding year.
    date['byear'] = date['bdate'].astype(str).str.split('-').apply(lambda x: x[0])

    for col in [''.join((x, '_pred')) for x in dates_sequence]:
        # Also keep the day and month predictions without casting NaT even if invalid
        date[col + '_day'] = date[col].transform(lambda x: np.nan if x == 'bad cpd' else x.split(':')[0])
        date[col + '_month'] = date[col].transform(lambda x: np.nan if x == 'bad cpd' else x.split(':')[1])
        date[col] = pd.to_datetime(date[col] + ':' + date['byear'], format='%d:%m:%Y', errors='coerce') # pylint: disable=C0301

    # Now, increase year if needed (i.e. december -> january switch).
    added_year = np.zeros(len(date))

    seq = ['bdate'] + [''.join((x, '_pred')) for x in dates_sequence]

    for idx in range(len(seq) - 1):
        var1, var2 = seq[idx], seq[idx + 1]

        is_less = date[var1] > date[var2]
        added_year += is_less
        date.loc[added_year > 0, var2] += pd.DateOffset(years=1)

    # Now we can flag sequence for consistency.
    flag_dates = verify_sequence_dates(date, seq)
    date['incons_date_seq'] = flag_dates == 0

    date = date.drop(columns=['bdate', 'byear'])

    # NOTE: Bad cpd are already replaced by nan, as they are not proper dates.

    return date


def _prepare_tab_b() -> pd.DataFrame:
    tab_b = pd.read_csv(FN_TAB_B_PRED, na_values=[''], keep_default_na=False)

    tab_b = tab_b.rename(columns={'journal': 'Filename'})
    tab_b = round_prob(tab_b, 2)
    tab_b.columns = [x.replace('-', '_') for x in tab_b.columns]

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    tab_b = tab_b.replace('bad cpd', np.nan)

    return tab_b


def _prepare_length() -> pd.DataFrame:
    length = pd.read_csv(FN_LENGTH_PRED, na_values=[''], keep_default_na=False)

    length = length.rename(columns={'journal': 'Filename'})
    length = round_prob(length, 2)

    flag_length = verify_sequence_nondecreasing(
        length, [''.join((x, '_pred')) for x in ['blength', 'length1y']]
        )
    length['incons_length_seq'] = flag_length == 0

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    length = length.replace('bad cpd', np.nan)

    return length


def _prepare_dabf() -> pd.DataFrame:
    dabf = pd.read_csv(FN_DABF_PRED, na_values=[''], keep_default_na=False)

    dabf = dabf.rename(columns={'journal': 'Filename'})
    dabf = round_prob(dabf, 2)

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    dabf = dabf.replace('bad cpd', np.nan)

    return dabf


def _prepare_preterm() -> pd.DataFrame:
    preterm = pd.read_csv(FN_PRETERM_PRED, na_values=[''], keep_default_na=False)

    preterm = preterm.rename(columns={'journal': 'Filename'})
    preterm = round_prob(preterm, 2)

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    preterm = preterm.replace('bad cpd', np.nan)

    return preterm


def _prepare_preterm_weeks() -> pd.DataFrame:
    preterm_weeks = pd.read_csv(FN_PRETERM_WEEKS_PRED, na_values=[''], keep_default_na=False)

    preterm_weeks = preterm_weeks.rename(columns={'journal': 'Filename'})
    preterm_weeks = round_prob(preterm_weeks, 2)

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    preterm_weeks = preterm_weeks.replace('bad cpd', np.nan)

    return preterm_weeks


def _prepare_bf7do() -> pd.DataFrame:
    bf7do = pd.read_csv(FN_BF7DO_PRED, na_values=[''], keep_default_na=False)

    bf7do = bf7do.rename(columns={'journal': 'Filename'})
    bf7do = round_prob(bf7do, 2)

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    bf7do = bf7do.replace('bad cpd', np.nan)

    return bf7do


def _prepare_nurse_lastname() -> pd.DataFrame:
    nurse_lastname = pd.read_csv(FN_NURSE_LASTNAME_PRED, na_values=[''], keep_default_na=False)

    nurse_lastname = nurse_lastname.rename(columns={'journal': 'Filename'})
    nurse_lastname = round_prob(nurse_lastname, 2)
    nurse_lastname.columns = [x.replace('-', '_') for x in nurse_lastname.columns]

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    nurse_lastname = nurse_lastname.replace('bad cpd', np.nan)

    prob_cols = [f'nurse_name_{i}_prob' for i in range(1, 4)]
    pred_cols = [f'nurse_name_{i}_pred' for i in range(1, 4)]
    assert nurse_lastname.shape[1] == len(prob_cols + pred_cols) + 1

    for i, prob_col in enumerate(prob_cols, start=1):
        nurse_lastname.rename(columns={prob_col: f'nn_ln_m_{i}_prob'}, inplace=True)

    for i, pred_col in enumerate(pred_cols, start=1):
        nurse_lastname[f'nn_ln_m_{i}_pred_cat'] = list(map(_create_name_category, nurse_lastname[pred_col])) # pylint: disable=C0301
        nurse_lastname[pred_col] = _names_to_numeric(nurse_lastname[pred_col].values, False)
        nurse_lastname.rename(columns={pred_col: f'nn_ln_m_{i}_pred'}, inplace=True)

    return nurse_lastname


def _prepare_nurse_firstname() -> pd.DataFrame:
    nurse_firstname = pd.read_csv(FN_NURSE_FIRSTNAME_PRED, na_values=[''], keep_default_na=False)

    nurse_firstname = nurse_firstname.rename(columns={'journal': 'Filename'})
    nurse_firstname = round_prob(nurse_firstname, 2)
    nurse_firstname.columns = [x.replace('-', '_') for x in nurse_firstname.columns]

    # Replace bad cpd with nan (as that is what it is for analysis purposes).
    nurse_firstname = nurse_firstname.replace('bad cpd', np.nan)

    prob_cols = [f'nurse_name_{i}_prob' for i in range(1, 4)]
    pred_cols = [f'nurse_name_{i}_pred' for i in range(1, 4)]
    assert nurse_firstname.shape[1] == len(prob_cols + pred_cols) + 1

    for i, prob_col in enumerate(prob_cols, start=1):
        nurse_firstname.rename(columns={prob_col: f'nn_fn_m_{i}_prob'}, inplace=True)

    for i, pred_col in enumerate(pred_cols, start=1):
        nurse_firstname[f'nn_fn_m_{i}_pred_cat'] = list(map(_create_name_category, nurse_firstname[pred_col])) # pylint: disable=C0301
        nurse_firstname[f'nn_fn_m_{i}_pred_ini'] = _names_to_numeric(nurse_firstname[pred_col].values, True) # pylint: disable=C0301
        nurse_firstname[pred_col] = _names_to_numeric(nurse_firstname[pred_col].values, False)
        nurse_firstname.rename(columns={pred_col: f'nn_fn_m_{i}_pred'}, inplace=True)

    return nurse_firstname


def _names_to_numeric(names: np.ndarray, only_initial: bool) -> list:
    '''
    Maps names to numeric values using hashing. Asserts no "collisions", i.e.
    same number of distincs cases post cast.

    '''
    assert isinstance(names, np.ndarray)
    assert isinstance(only_initial, bool)

    if only_initial:
        names = list(map(_name_to_initial, names))

    as_numeric = list(map(_name_to_numeric, names))

    assert len(set(as_numeric)) == len(set(names))

    return as_numeric


def _name_to_numeric(name: str) -> str:
    '''
    Hashes a name to an integer and keeps first 10 digits. For safety, appends
    a "key". Recall this is not in any way an encryption, but just a
    de-identifying exercise, and so this does not need to be safe in the sense
    of encryption. It is also one-way (i.e. a hash, not encryption).

    NaNs are kept as NaNs. "0=Mangler" is kept as "0=Mangler".

    '''
    max_str_length: int = 20

    if not isinstance(name, str) and math.isnan(name):
        return name

    assert isinstance(name, str)

    if name == '0=Mangler':
        return name

    assert len(name) <= max_str_length
    assert set(name).issubset(LETTERS_SET)

    hashed = hash(name + 'douglas adams')
    assert isinstance(hashed, int)

    if hashed < 0:
        hashed *= -1

    qhashed = str(hashed)[:10]

    return qhashed


def _name_to_initial(name: str) -> str:
    '''
    Maps a name to initial.

    NaNs are kept as NaNs. "0=Mangler" is kept as "0=Mangler".

    '''
    if not isinstance(name, str) and math.isnan(name):
        return name

    assert isinstance(name, str)
    assert len(name) > 0

    if name == '0=Mangler':
        return name

    return name[0]


def _create_name_category(name: str) -> str:
    if not isinstance(name, str) and math.isnan(name):
        return name

    if name == '0=Mangler':
        return name

    return 'id'


def verify_sequence_nondecreasing(data: pd.DataFrame, seq: list) -> pd.Series:
    """
    The sequence of e.g. weights is expected to be non-decreasing. This
    function finds a flag (array) of whether a sequence is "consistent", i.e.
    non-decreasing. Note that even if the seqauence is not "consistent", it
    might still be transcriped correctly (there exists such examples).

    Parameters
    ----------
    data : pd.DataFrame
        Data with e.g. weight predictions.
    seq : list
        The (ordered) list of the e.g. weight predictions.

    Returns
    -------
    flag : pd.Series
        Binary array/series, 1 = consistent, 0 = inconsistent.

    """
    flag = np.ones(len(data))

    for idx in range(len(seq) - 1):
        var1, var2 = seq[idx], seq[idx + 1]

        both_numeric = (data[var1].str.isnumeric()) & (data[var2].str.isnumeric())
        is_weakly_greater = pd.to_numeric(data.loc[both_numeric, var1]) <= pd.to_numeric(data.loc[both_numeric, var2]) # pylint: disable=C0301
        is_wrong = both_numeric
        is_wrong[is_weakly_greater[is_weakly_greater].index] = False

        flag = flag * (~is_wrong)

    return flag


def verify_sequence_dates(data: pd.DataFrame, seq: list) -> pd.Series:
    """
    The sequence of dates is expected to be increasing. This function finds a
    flag (array) of whether a sequence is "consistent", i.e. increasing.

    Parameters
    ----------
    data : pd.DataFrame
        Data with date predictions.
    seq : list
        The (ordered) list of the date predictions.

    Returns
    -------
    flag : pd.Series
        Binary array/series, 1 = consistent, 0 = inconsistent.

    """
    flag = np.ones(len(data))

    for idx in range(len(seq) - 1):
        var1, var2 = seq[idx], seq[idx + 1]

        is_greater_or_nan = ~(data[var2] < data[var1])

        flag = flag * is_greater_or_nan

    return flag


def round_prob(data: pd.DataFrame, nb_digits: int) -> pd.DataFrame:
    '''
    Modifies a pd.DataFrame to ensure all columns ending in "_prop" are rounded
    as requested.

    As a check, before rounding is applied it asserts a columns ending in
    "_pred" exists (i.e. the associated prediction for the probability).

    Parameters
    ----------
    data : pd.DataFrame
        Data to round specific columns of.
    nb_digits : int
        Number of digits to maintain.

    Returns
    -------
    data : pd.DataFrame
        Modified pd.DataFrame with rounded "_prob" columns.

    '''

    cols_to_round = [x for x in data.columns if x.endswith('_prob')]

    assert len(cols_to_round) > 0

    for col in cols_to_round:
        assert col.replace('_prob', '_pred') in data.columns
        data[col] = round(data[col], nb_digits)

    return data


def rename(fname: str) -> str:
    ''' Extract the filename from an entry in the "path" column, allowing to
    use it to merge with main data source.
    '''
    fname = fname.split('/')[-1]
    fname = fname.split('.page-')[0]

    return fname


def recode_array_str(array: np.ndarray) -> str:
    ''' Recodes the [[int], [int], [int]] entries to "int int int" entries.
    '''
    return ' '.join([str(x[0]) for x in array])


def drop_not_imported_entries(data_frame: pd.DataFrame) -> pd.DataFrame:
    ''' Drop all entries that stem from the NotImported folder, as they
    are all duplicates, see "./../checks/verify_notimported_duplicates.html".
    This is then run for both cluster and intensity data frames.
    '''
    find_notimported = lambda x: 'NotImported' in x
    from_notimported = pd.Series([find_notimported(x) for x in data_frame['path'].values])
    data_frame_sub = data_frame[~from_notimported]

    return data_frame_sub


def create_page_number_col(data_frame: pd.DataFrame) -> pd.Series:
    ''' From path column create page number column.
    '''
    find_page_number = lambda x: x[(x.find('page') + 5):(x.find('.jpg'))]
    page = pd.Series([find_page_number(x) for x in data_frame['path'].values])

    return page


def create_row_number(intensity_df):
    ''' Create row number col to intensity data frame.
    '''
    return intensity_df.groupby('path').cumcount() + 1


def load_prepare_merge(): # pylint: disable=R0914, R0912, R0915, C0116
    datasets = DATASETS

    assert len(datasets) == len(set(datasets.index))
    assert len(datasets) == len(set(datasets['loader']))

    main = datasets.loc['main', 'loader']()

    # CPR data
    main = main.merge(
        datasets.loc['cpr', 'loader'](),
        on=datasets.loc['cpr', 'merge-var'],
        how='left',
        )

    # Status data
    main = main.merge(
        datasets.loc['status', 'loader'](),
        on=datasets.loc['status', 'merge-var'],
        how='left',
        )

    # Cluster = treatment data
    main = main.merge(
        datasets.loc['cluster', 'loader'](),
        on=datasets.loc['cluster', 'merge-var'],
        how='left',
        )

    # Intensity = degree filled in data, treatment intensity
    main = main.merge(
        datasets.loc['intensity', 'loader'](),
        on=datasets.loc['intensity', 'merge-var'],
        how='left',
        )

    # Consider looping over all the rest, should be doable since consistent format

    # Weight data
    main = main.merge(
        datasets.loc['weight', 'loader'](),
        on=datasets.loc['weight', 'merge-var'],
        how='left',
        )

    # Date data
    main = main.merge(
        datasets.loc['date', 'loader'](),
        on=datasets.loc['date', 'merge-var'],
        how='left',
        )

    # Table B data
    main = main.merge(
        datasets.loc['tab-b', 'loader'](),
        on=datasets.loc['tab-b', 'merge-var'],
        how='left',
        )

    # Length
    main = main.merge(
        datasets.loc['length', 'loader'](),
        on=datasets.loc['length', 'merge-var'],
        how='left',
        )

    # Duration any breastfeeding
    main = main.merge(
        datasets.loc['dabf', 'loader'](),
        on=datasets.loc['dabf', 'merge-var'],
        how='left',
        )

    # Nurse lastname
    main = main.merge(
        datasets.loc['nurse-lastname', 'loader'](),
        on=datasets.loc['nurse-lastname', 'merge-var'],
        how='left',
        )

    # Nurse firstname
    main = main.merge(
        datasets.loc['nurse-firstname', 'loader'](),
        on=datasets.loc['nurse-firstname', 'merge-var'],
        how='left',
        )

    # Preterm
    main = main.merge(
        datasets.loc['preterm', 'loader'](),
        on=datasets.loc['preterm', 'merge-var'],
        how='left',
        )

    # Preterm (weeks)
    main = main.merge(
        datasets.loc['preterm-weeks', 'loader'](),
        on=datasets.loc['preterm-weeks', 'merge-var'],
        how='left',
        )

    # Breastfeeding/nutrition ~7-14 days old
    main = main.merge(
        datasets.loc['bf7do', 'loader'](),
        on=datasets.loc['bf7do', 'merge-var'],
        how='left',
        )

    # Drop Filename since we are done merging and it is an ID var
    main = main.drop(columns='Filename')

    # Duplicate check
    for col in main.columns:
        try:
            vals, counts = np.unique(main[col].dropna(), return_counts=True)
        except TypeError as err:
            print(f'Duplicate check does not work for column {col}, error =', err)
            continue

        if min(counts) == 1:
            print(f'Unique entries in column {col}!')

            if col not in ('Id', 'Filename', 'cpr'):
                print(pd.DataFrame({'vals': vals, 'counts': counts}))

    # Stricter check where we drop duplicates over Id and then same as
    # above, since now there are many that are "cloned" which may be cause of
    # no problems with uniques
    # Obviously many uniques now - but is it a problem?
    # Not currently. To see, use variable explorer on look_through

    submain = main.drop_duplicates('Id')
    look_through = dict()

    for col in submain.columns:
        try:
            vals, counts = np.unique(submain[col].dropna(), return_counts=True)
        except TypeError as err:
            print(f'Duplicate check does not work for column {col}, error =', err)
            continue

        if min(counts) == 1:
            print(f'({sum(counts == 1)}) Unique entries in column {col}!')

            if col not in ('Id', 'Filename', 'cpr'):
                # print(pd.DataFrame({'vals': vals, 'counts': counts}))
                look_through[col] = pd.DataFrame({'vals': vals, 'counts': counts})

    # Save...
    fn_out = os.path.join(
        'Y:/RegionH/Scripts/users/tsdj/data_to_dst/',
        '-'.join((datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S'), 'to-dst.csv')),
        )

    assert not os.path.isfile(fn_out)
    print(f'Writing file {fn_out}!')
    main.to_csv(fn_out, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Reload and ensure still looks fine
    dtypes_map = dict(zip(main.dtypes.index, main.dtypes.values))
    parse_dates = [k for k, v in dtypes_map.items() if v == np.dtype('<M8[ns]')]
    for key in parse_dates:
        del dtypes_map[key]

    main_reloaded = pd.read_csv(fn_out, dtype=dtypes_map, parse_dates=parse_dates)

    assert all(main.columns == main_reloaded.columns)

    for col in main.columns:
        if col in ['empty2', 'empty1', '36mon', '30mon', '24mon', '18mon', '15mon']:
            # These cols are problematic to check equality for due to mixed
            # data types.
            not_eq = ~(main[col].dropna() == main_reloaded[col].dropna()) # pylint: disable=E1136
            assert main[col].dropna()[not_eq].astype(str).equals(main_reloaded[col].dropna()[not_eq].astype(str)) # pylint: disable=C0301, E1136
            continue

        if main[col].equals(main_reloaded[col]): # pylint: disable=E1136
            continue

        if main[col].dtype == np.dtype('O'):
            if main[col].equals(main_reloaded[col].fillna('')): # pylint: disable=E1136
                # Handles empty strings converted to nans
                continue
            if main[col].dropna().astype(str).equals(main_reloaded[col].dropna()): # pylint: disable=E1136
                # Handles bools converted to strings for example
                continue

        raise Exception(col, main[col], main_reloaded[col]) # pylint: disable=E1136

    # Check balance across those with long tables and those without
    # main['longtable'] = main['type'].isin({4, 5, 25, 27})
    # main['has_longtable'] = main.groupby('Id')['longtable'].transform(lambda x: sum(x) > 0)
    # sub = main.drop_duplicates('Id')

    # pred_cols = [x for x in sub.columns if x[-5:] == '_pred']
    # balance_df = sub.groupby('has_longtable')[pred_cols].apply(lambda x: x.notnull().mean()).T
    # balance_df['diff'] = balance_df[True] - balance_df[False]

    # Check with old data
    # main_old = pd.read_csv(
    #     r'Y:\RegionH\Scripts\users\tsdj\data_to_dst\210105_upload.csv',
    #     dtype=dtypes_map, parse_dates=parse_dates,
    #     )

    # look_through_2 = {}
    # for col in main_old.columns:
    #     print(col)
    #     if not main_reloaded[col].equals(main_old[col]):
    #         look_through_2[col] = pd.DataFrame({'old': main_old[col], 'new': main_reloaded[col]})
    #         look_through_2[col] = look_through_2[col].fillna('NaN-PlaceHolder')
    #         look_through_2[col]['eq'] = look_through_2[col]['old'] == look_through_2[col]['new']
    #         look_through_2[col] = look_through_2[col][~look_through_2[col]['eq']]
    #         look_through_2[col] = look_through_2[col][['old', 'new']]

    # {k: len(v) for k, v in look_through_2.items()}


if __name__ == '__main__':
    DATASETS = pd.DataFrame(
        {
            'loader': [
                _prepare_main, _prepare_cpr, _prepare_status, _prepare_cluster,
                _prepare_intensity, _prepare_weight, _prepare_date, _prepare_tab_b,
                _prepare_length, _prepare_dabf, _prepare_nurse_lastname,
                _prepare_nurse_firstname,
                _prepare_preterm, _prepare_preterm_weeks, _prepare_bf7do,
                ],
            'merge-var': [
                None, 'Id', 'Id', 'Filename', ['Filename', 'page'], 'Filename',
                'Filename', 'Filename', 'Filename', 'Filename', 'Filename',
                'Filename', 'Filename', 'Filename', 'Filename',
                ]
            },
        index=[
            'main', 'cpr', 'status', 'cluster', 'intensity',
            'weight', 'date', 'tab-b', 'length', 'dabf', 'nurse-lastname',
            'nurse-firstname', 'preterm', 'preterm-weeks', 'bf7do',
            ],
        )

    print('READ CAREFULLY IF BELOW SETTINGS ARE CORRECT!\n')
    print(DATASETS)

    # load_prepare_merge()
