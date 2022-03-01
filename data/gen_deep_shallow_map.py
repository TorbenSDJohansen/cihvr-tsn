# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:15:57 2020

@author: tsdj

The script creates mappings between a flat file format and deep file format and
between the journals (with multiple pages) and the images (each of one page).

This is highly useful as the deep format is unneeded, and thus this allows for
the creation of a shallow format, useful when storing minipics and more.

This is a "run-once" type of script! It just generates files to read from.

STATUS (2020-06-19):
    It contains some hastagged lines that are a nice check. So not sure whether
    to delete (it is checked after all) or just let it stay - currently left.
"""

import sys
import os
import pickle

# import pandas as pd
import numpy as np

def _dig_for_ftype(directory: str, ftype: str) -> list:
    """
    Recursively digs through folders for a specific file type, storing the path
    to the file and the file name in a list (of strings).

    Parameters
    ----------
    directory : str
        The directory to search in, adding all files of the specified type to
        the list and continuing searching in all folders.
    ftype : str
        The file type to add (such as '.jpg'). MUST be lower case (as the file
        is always cast to lower()).

    Returns
    -------
    list
        A list of all the specified files in the directory/child directories,
        with their full path.

    """
    items = os.listdir(directory)
    files = []

    for item in items:
        if len(item) > len(ftype) and item[-len(ftype):].lower() == ftype:
            files.append('/'.join((directory, item)))
        elif os.path.isdir('/'.join((directory, item))):
            files.extend(_dig_for_ftype('/'.join((directory, item)), ftype))
        else:
            pass

    return files


def _search_deep(root: str, include: list or tuple, ftype: str) -> list:
    """
    Recursively digs through folders for a specific file type, storing the path
    to the file and the file name in a list (of strings). Only difference to
    `_dig_for_ftype` is that the start is now a directory where only the
    folders included (specified by `include`) are searched in.

    Parameters
    ----------
    root : str
        The "root" directory to start the search in.
    include : list or tuple
        The subfolders of `root` to search in.
    ftype : str
        The file type to add (such as '.jpg'). MUST be lower case (as the file
        is always cast to lower()).

    Returns
    -------
    list
        A list of all the specified files in the included child directories,
        with their full path.

    """
    files = []

    for subdir in (''.join((root, folder)) for folder in include):
        files.extend(_dig_for_ftype(subdir, ftype))

    return files


def gen_maps():
    """
    The function generates maps (dictionaries) between different files or
    between the same file with/without its path.

    Notation (meaning of suffixes):
        _ds = DEEP -> SHALLOW
        _sd = SHALLOW -> DEEP
        _ss = SHALLOW -> SHALLOW
        _dd = DEEP -> DEEP

    Raises
    ------
    Exception
        This function relies on ordered dictionaries, and thus require Python
        >= 3.6. If not met, an exception is raised.

    Parameters
    ----------
    None

    Returns
    -------
    None.

    """
    if sys.version_info[0] < 3 or sys.version_info[1] < 6:
        raise Exception(
            f'''
            This function relies on ordered dictionaries!
            Use a newer Python version (>= 3.6)!
            You are using version {sys.version}!
            '''
            )

    # fn_main = 'Y:/RegionH/SPJ/Database/export_181106.txt'
    # main_df = pd.read_csv(fn_main, sep=";")

    root_journals = 'Y:/RegionH/SPJ/Journals/'
    root_images = 'Y:/RegionH/SPJ/Journals_jpg/'
    include = (
        '1959', '1960', '1961', '1962', '1963',
        '1964', '1965', '1966', '1967', 'Extra',
        )
    journal_files = _search_deep(root_journals, include, '.pdf')
    image_files = _search_deep(root_images, include, '.jpg')

    map_journals_ds = {file: file.split('/')[-1] for file in journal_files}
    map_images_ds = {file: file.split('/')[-1] for file in image_files}

    assert len(map_journals_ds) == len(set(map_journals_ds.values()))
    assert len(map_images_ds) == len(set(map_images_ds.values()))

    map_journals_sd = {value: key for key, value in map_journals_ds.items()}
    map_images_sd = {value: key for key, value in map_images_ds.items()}

    # Good match between data frame and files. See below:
    # set(map_journals_sd.keys()) - set(main_df['Filename'].values)
    # It outputs
    # >>> {'SP2_22407.pdf'}
    # But this file is also the 2 blue pages with DØD.
    # And noteably
    # set(main_df['Filename'].values) - set(map_journals_sd.keys())
    # >>> set()

    # Map journal to its pages
    pairs = [x.split('.page-') for x in map_images_sd.keys()]
    pairs = [[x[0], int(x[1].split('.jpg')[0])] for x in pairs]
    journals = np.array([x[0] for x in pairs])

    # Below is SLOW, but as this is a run-once file no reason to optimize.
    indices = []
    for i, journal in enumerate(map_journals_sd.keys()):
        if i % 5000 == 0:
            print(f'Progress: {round(i / len(map_journals_sd.keys()), 2) * 100}%.')
        indices.append(np.where(journals == journal)[0])

    # dict of ss, then easy to construct all dd, sd, ds combinations by looking
    # in other dicts
    map_journals_images_ss = dict()
    image_files_s = list(map_images_sd.keys())

    for i, journal in enumerate(map_journals_sd.keys()):
        value = [image_files_s[idx] for idx in indices[i]]
        map_journals_images_ss[journal] = value

    assert len(map_journals_images_ss) == len(map_journals_sd.keys())
    assert len(map_images_sd) == sum([len(x) for x in map_journals_images_ss.values()])

    map_journals_images_dd = dict()

    for key, value in map_journals_images_ss.items():
        key_d = map_journals_sd[key]
        value_d = [map_images_sd[subvalue] for subvalue in value]
        map_journals_images_dd[key_d] = value_d

    assert len(map_journals_images_dd) == len(map_journals_ds.keys())
    assert len(map_images_ds) == sum([len(x) for x in map_journals_images_dd.values()])

    dump_root = 'Y:/RegionH/Scripts/users/tsdj/storage/maps/'
    pickle.dump(map_journals_ds, open(''.join((dump_root, 'map_journals_ds.pkl')), 'wb'))
    pickle.dump(map_images_ds, open(''.join((dump_root, 'map_images_ds.pkl')), 'wb'))
    pickle.dump(map_journals_sd, open(''.join((dump_root, 'map_journals_sd.pkl')), 'wb'))
    pickle.dump(map_images_sd, open(''.join((dump_root, 'map_images_sd.pkl')), 'wb'))
    pickle.dump(map_journals_images_ss, open(''.join((dump_root, 'map_journals_images_ss.pkl')), 'wb'))
    pickle.dump(map_journals_images_dd, open(''.join((dump_root, 'map_journals_images_dd.pkl')), 'wb'))


if __name__ == '__main__':
    gen_maps()
