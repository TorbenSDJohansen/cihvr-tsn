# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:44:08 2022

@author: sa-tsdj
"""


import os

import pandas as pd


class Counter:
    def __init__(self):
        self.counts = {}
        self.journals = {}

    def additem(self, nb_pages: int, journal: str):
        if nb_pages not in self.counts:
            self.counts[nb_pages] = 1
            self.journals[nb_pages] = [journal]
        else:
            self.counts[nb_pages] += 1
            self.journals[nb_pages].append(journal)


def main():
    root_pdf = r'Y:\RegionH\SPJ\Journals'
    root_jpg = r'Y:\RegionH\SPJ\Journals_jpg'

    # Recall NotImported are duplicates
    folders = [*(str(i) for i in range(1959, 1968)), 'Extra']

    counter = Counter()

    for folder in folders:
        print(f'Working on folder {folder}.')

        folder_pdf = os.path.join(root_pdf, folder)
        folder_jpg = os.path.join(root_jpg, folder)

        for subdir in os.listdir(folder_pdf):
            subdir_pdf = os.path.join(folder_pdf, subdir)
            subdir_jpg = os.path.join(folder_jpg, subdir)

            if not os.path.isdir(subdir_pdf):
                continue

            journals = os.listdir(subdir_pdf)
            pages = os.listdir(subdir_jpg)

            for journal in journals:
                if not journal.lower().endswith('pdf'):
                    continue

                nb_pages = len([x for x in pages if x.startswith(journal)])
                counter.additem(nb_pages, journal)

        print(f'Current counts: {counter.counts}.')

    counts = pd.DataFrame(counter.counts, index=['Nb. journals']).T.sort_index()
    counts.to_csv(r'C:\Users\sa-tsdj\Desktop\nb_pages.csv')

    # searched_journals = []
    # for journals in counter.journals.values():
    #     searched_journals.extend(journals)
