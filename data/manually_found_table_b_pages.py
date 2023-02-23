# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Candidates are found by considering pages with large determinant of transfor-
mation matrix.

Can be extended to also very low determinants etc.
"""


import pickle

import cv2

def show(image_file: str or list or tuple):
    if isinstance(image_file, (list, tuple)):
        for i, fname in enumerate(image_file):
            image = cv2.imread(fname)
            cv2.namedWindow(f'tmp{i}', cv2.WINDOW_NORMAL)
            cv2.imshow(f'tmp{i}', image)
    else:
        image = cv2.imread(image_file)
        cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
        cv2.imshow('tmp', image)

    cv2.waitKey()
    cv2.destroyAllWindows()


FN_MAP_JOURNALS_IMAGES_SS = r'Y:\RegionH\Scripts\users\tsdj\storage\maps\map_journals_images_ss.pkl'
FN_MAP_IMAGES_SD = r'Y:\RegionH\Scripts\users\tsdj\storage\maps\map_images_sd.pkl'

with open(FN_MAP_JOURNALS_IMAGES_SS, 'rb') as file:
    MAP_JOURNALS_IMAGES_SS = pickle.load(file)

with open(FN_MAP_IMAGES_SD, 'rb') as file:
    MAP_IMAGES_SD = pickle.load(file)

TO_REVIEW_1 = [
    'SP2_07945.pdf',
    'SPJ_2014-07-04_0377.PDF',
    'SPJ_2014-06-20_0112.PDF',
    'SP2_07593.pdf',
    'SP4_05116.pdf',
    'SP4_18350.pdf',
    'SP2_07872.pdf',
    'SPJ_2014-07-08_0153.PDF',
    'SPJ_2014-06-20_0128.PDF',
    'SP4_07579.pdf',
    'SPJ_2014-06-20_0125.PDF',
    'SPJ_2014-05-13_0451.PDF',
    'SP4_13349.pdf',
    'SP4_03316.pdf',
    'SP4_22339.pdf',
    'SPJ_2014-04-23_0371.PDF',
    'SPJ_2014-06-20_0115.PDF',
    'SPJ_2014-04-11_0017.PDF',
    'SPJ_2014-06-20_0119.PDF',
    'SPJ_2014-04-04_0114.PDF',
    'SPJ_2014-11-07_0026.PDF',
    'SPJ_2014-11-10_0068.PDF',
    'SPJ_2014-07-17_0575.PDF',
    'SPJ_2014-07-17_0586.PDF',
    'SPJ_2014-11-07_0042.PDF',
    'SPJ_2014-07-09_0530.PDF',
    'SPJ_2014-05-23_0140.PDF',
    'SPJ_2014-04-25_0399.PDF',
    'SPJ_2014-07-17_0603.PDF',
    'SP4_03939.pdf',
    'SPJ_2014-07-07_0339.PDF',
    'SP4_21073.pdf',
    'SPJ_2014-05-20_0158.PDF',
    'SPJ_2014-07-17_0577.PDF',
    'SPJ_2014-04-23_0364.PDF',
    'SPJ_2014-07-17_0584.PDF',
    'SPJ_2014-06-06_0727.PDF',
    'SPJ_2014-07-08_0309.PDF',
    'SPJ_2014-07-17_0579.PDF',
    'SPJ_2014-07-03_0056.PDF',
    'SPJ_2014-04-28_0026.PDF',
    'SP6_00309.pdf',
    'SPJ_2014-11-07_0145.PDF',
    'SP6_00007.pdf',
    'SP4_21310.pdf',
    'SPJ_2014-04-30_0188.PDF',
    'SPJ_2014-06-20_0567.PDF',
    'SPJ_2014-05-05_0521.PDF',
    'SPJ_2014-06-06_0768.PDF',
    'SPJ_2014-04-10_0057.PDF',
    'SP6_02273.pdf',
    'SPJ_2014-07-11_0066.PDF',
    'SPJ_2014-07-04_0519.PDF',
    'SPJ_2014-06-19_0424.PDF',
    'SPJ_2014-11-07_0084.PDF',
    'SPJ_2014-04-01_0186.PDF',
    'SPJ_2014-04-23_0375.PDF',
    'SPJ_2014-04-23_0384.PDF',
    'SPJ_2014-04-23_0373.PDF',
    'SPJ_2014-04-04_0112.PDF',
    'SPJ_2014-04-01_0187.PDF',
    'SPJ_2014-04-23_0366.PDF',
    'SPJ_2014-04-04_0138.PDF',
    'SPJ_2014-04-25_0398.PDF',
    'SPJ_2014-04-03_0131.PDF',
    'SP4_02374.pdf',
    'SPJ_2014-04-23_0378.PDF',
    'SPJ_2014-04-04_0113.PDF',
    'SPJ_2014-05-20_0015.PDF',
    'SPJ_2014-04-02_0230.PDF',
    'SPJ_2014-04-10_0037.PDF',
    'SPJ_2014-04-02_0232.PDF',
    'SP4_06992.pdf',
    'SPJ_2014-04-02_0188.PDF',
    'SPJ_2014-04-04_0148.PDF',
    'SPJ_2014-05-12_0331.PDF',
    'SP2_15679.pdf',
    'SPJ_2014-04-04_0115.PDF',
    'SPJ_2014-04-02_0197.PDF',
    'SPJ_2014-04-28_0014.PDF',
    'SP2_19275.pdf',
    'SPJ_2014-07-17_0581.PDF',
    'SPJ_2014-04-02_0184.PDF',
    'SPJ_2014-04-23_0383.PDF',
    'SP4_03682.pdf',
    'SP4_05052.pdf',
    'SPJ_2014-04-30_0241.PDF',
    'SPJ_2014-04-02_0186.PDF',
    'SP4_03940.pdf',
    'SP4_03941.pdf',
    'SPJ_2014-07-24_0020.PDF',
    'SPJ_2014-07-23_0669.PDF',
    'SP6_00941.pdf',
    ]
INPUTS_1 = [
    '0',
    '0',
    '1',
    '0',
    '-1',
    '2',
    '0',
    '-1',
    '1',
    '-1',
    '1',
    '3',
    '2',
    '-1',
    '2',
    '-1',
    '1',
    '1',
    '1',
    '2',
    '2',
    '-1',
    '-1',
    '-1',
    '1',
    '1',
    '-1',
    '-1',
    '-1',
    '2',
    '-1',
    '2',
    '2',
    '-1',
    '-1',
    '-1',
    '-1',
    '-1',
    '-1',
    '1',
    '-1',
    '2',
    '1',
    '-1',
    '-1',
    '-1',
    '-1',
    '-1',
    '-1',
    '2',
    '2',
    '-1',
    '1',
    '-1',
    '-1',
    '-1',
    '-1',
    '-1',
    '-1',
    '2',
    '-1',
    '-1',
    '-1',
    '2',
    '2',
    '2',
    '-1',
    '2',
    '-1',
    '-1',
    '2',
    '3',
    '-1',
    '-1',
    '-1',
    '-1',
    '-2',
    '2',
    '-1',
    '2',
    '-2',
    '-1',
    '-1',
    '2',
    '2',
    '2',
    '-1',
    '-1',
    '2',
    '2',
    '-1',
    '-1',
    '-1'
    ]

PACKAGES = [
    {'journals': TO_REVIEW_1, 'pages': INPUTS_1},
    ]
ADDITIONAL_TABLE_B_IMAGES = []

for package in PACKAGES:
    ADDITIONAL_TABLE_B_IMAGES.extend([
        ''.join((x, '.page-', y, '.jpg')) for x, y in zip(package['journals'], package['pages'])
        if int(y) > 0
        ])


def main(to_review):
    inputs = []

    for i, journal in enumerate(to_review):
        print(f'Showing journal {i + 1} of {len(to_review)}: "{journal}".')
        candidates = MAP_JOURNALS_IMAGES_SS[journal]
        candidates = [MAP_IMAGES_SD[x] for x in candidates]
        show(candidates)
        inputs.append(input())

    return inputs


if __name__ == '__main__':
    INPUTS = main(TO_REVIEW_1)
