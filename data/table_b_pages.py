# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import pickle

from typing import Dict

from prepare_data_dst import _prepare_status, _prepare_main
from manually_found_table_b_pages import ADDITIONAL_TABLE_B_IMAGES


def get_table_b_pages() -> Dict[str, str]:
    '''
    Returns a dictionary with (key, value)-pairs like:

        ('Y:/RegionH/SPJ/Journals_jpg/1959/2014-03-31/SPJ_2014-03-31_0002.PDF.page-0.jpg', 'SPJ_2014-03-31_0002.PDF.page-0.jpg')

    where the key is the image file with the full path and the value is the
    basename of the image file.

    This is useful to iterate over and use the KEY to load the image and the
    VALUE as the name of the images of the segmented fields from the image.

    '''
    fn_map_images_ds = 'Y:/RegionH/Scripts/users/tsdj/storage/maps/map_images_ds.pkl'

    with open(fn_map_images_ds, 'rb') as file:
        map_images_ds = pickle.load(file)

    status = _prepare_status()
    main_ds = _prepare_main()

    merged = status.merge(main_ds, how='inner', on='Id')[['Id', 'Filename', 'exclude']]
    bad_cases = merged[merged['exclude'].isin(('Non-record',))]['Filename']

    map_images_ds_sub = {
        k: v for k, v in map_images_ds.items()
        if '.page-0.' in k or v in set(ADDITIONAL_TABLE_B_IMAGES)
        }

    map_images_ds_sub = {
        k: v for k, v in map_images_ds_sub.items()
        if v[:-11] not in bad_cases.values or v in set(ADDITIONAL_TABLE_B_IMAGES)
        }

    return map_images_ds_sub
