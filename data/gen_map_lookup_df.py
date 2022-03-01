# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:38:02 2020

@author: tsdj

STATUS:
    Finalized for the time, but may be extended as additional columns/cells
    are requested.
"""

import pickle

def _run():
    """
    Creates and saves a mapping between the names used in used in the CIHVR
    data and the names used in this "project" for folder, files etc.

    This is used to map between the lookups used "internally" and the external
    data, such as in `gen_labels.py`.

    Parameters
    ----------
    None

    Returns
    -------
    None.

    """
    # Need to know the column name in the data frame - both in order to extract
    # the training data and in order to fill it back in.
    map_lookup_df = {
        # Yes, they do count from 2 - since 1 refers to a number not in Table B
        'weight-0-mo': 'bweight',
        'weight-1-mo': 'weightv2',
        'weight-2-mo': 'weightv3',
        'weight-3-mo': 'weightv4',
        'weight-4-mo': 'weightv5',
        'weight-6-mo': 'weightv6',
        'weight-9-mo': 'weightv7',
        'weight-12-mo': 'weightv8',
        # 'date-0-mo', # I dont think this exists, but in can be created from
        # the other variable we have, i.e. dob, mob, yob, even CPR.
        'date-1-mo': 'datev2',
        'date-2-mo': 'datev3',
        'date-3-mo': 'datev4',
        'date-4-mo': 'datev5',
        'date-6-mo': 'datev6',
        'date-9-mo': 'datev7',
        'date-12-mo': 'datev8',
        'length-0-mo': 'blength',
        'length-12-mo': 'length1y',
        # table B small cells
        'tab-b-c1-1-mo': 'economyv2',
        'tab-b-c2-1-mo': 'harmonyv2',
        'tab-b-c3-1-mo': 'm_men_capv2',
        'tab-b-c4-1-mo': 'm_phys_capv2',
        'tab-b-c5-1-mo': 'homeworkv1',
        'tab-b-c6-1-mo': 'Work_outs_v1',
        'tab-b-c7-1-mo': 'daycarev1',
        'tab-b-c8-1-mo': 'carev1',
        'tab-b-c1-6-mo': 'economyv6',
        'tab-b-c2-6-mo': 'Harmonyv6',
        'tab-b-c3-6-mo': 'm_men_capv6',
        'tab-b-c4-6-mo': 'm_phys_capv6',
        'tab-b-c5-6-mo': 'homeworkv6',
        'tab-b-c6-6-mo': 'work_outs_v6',
        'tab-b-c7-6-mo': 'daycarev6',
        'tab-b-c8-6-mo': 'carev6',
        'tab-b-c15-1-mo': 'bvf2',
        'tab-b-c15-2-mo': 'bvf3',
        'tab-b-c15-3-mo': 'bvf4',
        'tab-b-c15-4-mo': 'bvf5',
        'tab-b-c15-6-mo': 'bvf6',
        'tab-b-c15-9-mo': 'bvf7',
        'tab-b-c15-12-mo': 'bvf8',
        # various
        'dura-any-breastfeed': 'bfdurany',
        # nurse names. note these are named by tsdj, see the following script
        # `~/code/data/prepare_nuse_name_data.py`
        'nurse-name-1': 'nurse-name-1',
        'nurse-name-2': 'nurse-name-2',
        'nurse-name-3': 'nurse-name-3',
        }

    assert len(map_lookup_df) == len(set(map_lookup_df.values()))

    pickle.dump(
        map_lookup_df,
        open('Y:/RegionH/Scripts/users/tsdj/storage/maps/map_lookup_df.pkl', 'wb'),
        )


if __name__ == '__main__':
    _run()
