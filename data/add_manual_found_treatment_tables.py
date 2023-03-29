# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

During updated segmentation, some Treatment Table pages were found. Check if
they overlap with current predictions and if not add them to list of Treatment
Table pages.

RESULT: All new candidates are already covered.

"""


import os

from prepare_data_dst import _prepare_cluster


PAGE_0_TREATMENT_TABLES = [
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-01/SPJ_2014-04-01_0186.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-01/SPJ_2014-04-01_0187.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-02/SPJ_2014-04-02_0184.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-02/SPJ_2014-04-02_0188.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-02/SPJ_2014-04-02_0197.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-02/SPJ_2014-04-02_0232.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-03/SPJ_2014-04-03_0131.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-04/SPJ_2014-04-04_0113.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-04/SPJ_2014-04-04_0114.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-04/SPJ_2014-04-04_0115.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-04/SPJ_2014-04-04_0138.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-04/SPJ_2014-04-04_0148.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-10/SPJ_2014-04-10_0037.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-10/SPJ_2014-04-10_0057.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-23/SPJ_2014-04-23_0364.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-23/SPJ_2014-04-23_0366.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-23/SPJ_2014-04-23_0371.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-23/SPJ_2014-04-23_0378.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-23/SPJ_2014-04-23_0383.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-23/SPJ_2014-04-23_0384.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-25/SPJ_2014-04-25_0398.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-25/SPJ_2014-04-25_0399.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-28/SPJ_2014-04-28_0014.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-30/SPJ_2014-04-30_0188.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-04-30/SPJ_2014-04-30_0241.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-05-05/SPJ_2014-05-05_0521.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-05-12/SPJ_2014-05-12_0331.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-05-20/SPJ_2014-05-20_0015.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1959/2014-05-20/SPJ_2014-05-20_0158.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1960/2014-05-23/SPJ_2014-05-23_0140.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1960/2014-06-06/SPJ_2014-06-06_0727.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1960/2014-06-19/SPJ_2014-06-19_0424.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1960/2014-06-20/SPJ_2014-06-20_0567.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-07/SPJ_2014-07-07_0339.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-17/SPJ_2014-07-17_0575.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-17/SPJ_2014-07-17_0577.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-17/SPJ_2014-07-17_0579.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-17/SPJ_2014-07-17_0581.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-17/SPJ_2014-07-17_0584.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-17/SPJ_2014-07-17_0586.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-17/SPJ_2014-07-17_0603.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-23/SPJ_2014-07-23_0669.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1961/2014-07-24/SPJ_2014-07-24_0020.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1962/2014-09-02/SP4_01546.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1962/2014-09-09/SP4_02374.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1962/2014-09-12/SP6_00309.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1963/2014-09-22/SP4_03682.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1963/2014-09-23/SP4_03939.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1963/2014-09-23/SP4_03941.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1964/2014-11-10/SPJ_2014-11-10_0068.PDF.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/1967/2015-03-02/SP8_00277.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/Extra/2015-04-09/SP4_22339.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/Extra/2015-04-15/SP2_39482.pdf.page-0.jpg',
    'Y:/RegionH/SPJ/Journals_jpg/Extra/2015-04-15/SP2_39483.pdf.page-0.jpg',
    ]


def main():
    cluster = _prepare_cluster()

    cluster['longtable'] = cluster['type'].isin({4, 5, 25, 27})
    cluster['has_longtable'] = cluster.groupby('Filename')['longtable'].transform(lambda x: sum(x) > 0)

    sub = cluster[cluster['has_longtable']]

    current_treatment_table_journals = set(sub['Filename'])
    new_candidate_treatment_table_journals = set(os.path.basename(x).split('.page')[0] for x in PAGE_0_TREATMENT_TABLES)
    assert len(new_candidate_treatment_table_journals) == len(PAGE_0_TREATMENT_TABLES)

    new_treatment_table_journals = new_candidate_treatment_table_journals - current_treatment_table_journals

    # Double check: (1) No new candidates are not already covered. (2) To check
    # this is not due to differing formats, check also overlap is same as new
    # candidate files.

    assert new_treatment_table_journals == set()
    assert new_candidate_treatment_table_journals & current_treatment_table_journals == new_candidate_treatment_table_journals


if __name__ == '__main__':
    main()
