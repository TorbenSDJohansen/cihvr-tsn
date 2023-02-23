# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

"""


import pickle

from typing import Tuple, Dict

import PIL

import pandas as pd

from matplotlib import pyplot as plt

from manually_found_table_b_pages import ADDITIONAL_TABLE_B_IMAGES


FN_CLUSTER = 'Y:/RegionH/Scripts/data/treatment_pages_classifications.pkl'

def rename(fname: str) -> str:
    ''' Extract the filename from an entry in the "path" column, allowing to
    use it to merge with main data source.
    '''
    fname = fname.split('/')[-1]
    fname = fname.split('.page-')[0]

    return fname


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


def load() -> Tuple[pd.DataFrame, Dict[str, str]]:
    fn_df_main = 'Y:/RegionH/SPJ/Database/export_181106.txt'
    fn_map_ds = 'Y:/RegionH/Scripts/users/tsdj/storage/maps/map_images_ds.pkl'

    df_main = pd.read_table(fn_df_main, sep=';')
    df_main = df_main.drop_duplicates('Filename')

    cluster = _prepare_cluster()

    df_main = df_main.merge(
        cluster,
        on='Filename',
        how='left',
        )

    df_main['longtable'] = df_main['type'].isin({4, 5, 25, 27})
    df_main['has_longtable'] = df_main.groupby('Id')['longtable'].transform(lambda x: sum(x) > 0)
    df_main = df_main.drop_duplicates('Filename')

    with open(fn_map_ds, 'rb') as file:
        map_images_ds = pickle.load(file)

    map_images_ds = {
        k: v for k, v in map_images_ds.items()
        if '.page-0.' in k or v in set(ADDITIONAL_TABLE_B_IMAGES)
        }

    return df_main, map_images_ds


def main():
    df_main, map_images_ds = load()

    map_file_dob = dict(df_main[['Filename', 'BirthDateDay']].values)
    map_file_yob = dict(df_main[['Filename', 'BirthDateYear']].values)
    map_file_has_long_table = dict(df_main[['Filename', 'has_longtable']].values)

    _stats = []

    for i, (fn_image, fn_image_short) in enumerate(map_images_ds.items(), start=1):
        if i % 1000 == 0:
            print(f'{i} of {len(map_images_ds)}')

        fn_image_short = fn_image_short.split('.page-')[0]
        dob = map_file_dob.get(fn_image_short, None)
        yob = map_file_yob.get(fn_image_short, None)
        has_long_table = map_file_has_long_table.get(fn_image_short, None)

        if dob is None or yob is None or has_long_table is None:
            continue

        image = PIL.Image.open(fn_image)

        _stats.append((fn_image_short, dob, yob, has_long_table, *image.size))

    stats = pd.DataFrame(_stats, columns=['file', 'dob', 'yob', 'has_long_table', 'width', 'height'])
    sub = stats[~stats['dob'].isnull()]

    sub_born13 = sub[sub['dob'].isin((1, 2, 3))]
    sub_control = sub[~sub['dob'].isin((1, 2, 3))]
    sub_has_long_table = sub[sub['has_long_table']]

    plt.hist(sub_born13['height'], alpha=0.5, label='Born 1-3')
    plt.hist(sub_has_long_table['height'], alpha=0.5, label='Treatment Table')
    plt.hist(sub_control['height'], alpha=0.5, label='Born 4-31')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Number of journals')
    plt.legend()
    plt.savefig(r'Y:\RegionH\Scripts\users\tsdj\storage\results\image-height-by-group-v1.png')
    plt.show()

    plt.hist(sub_born13['height'], alpha=0.5, label='Born 1-3')
    plt.hist(sub_has_long_table['height'], alpha=0.5, label='Treatment Table')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Number of journals')
    plt.legend()
    plt.savefig(r'Y:\RegionH\Scripts\users\tsdj\storage\results\image-height-by-group-v2.png')
    plt.show()

    for yob in range(59, 68):
        plt.hist(sub_born13.loc[sub_born13['yob'] == yob, 'height'], alpha=0.5, label=str(yob))

    plt.xlabel('Height (pixels)')
    plt.ylabel('Number of journals')
    plt.savefig(r'Y:\RegionH\Scripts\users\tsdj\storage\results\image-height-by-year-born13.png')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
