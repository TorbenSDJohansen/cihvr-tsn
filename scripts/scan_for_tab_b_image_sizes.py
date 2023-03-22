# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Select SP0_00003 for all 112 tab-b-c{1, 2, .., 16}-{1, 2, 3, 4, 6, 9, 12}-mo
fields to get image size for all fields for Type A (most common type).

"""


import os

from PIL import Image

import pandas as pd


def main():
    root = r'Y:\RegionH\Scripts\data\storage\minipics\TypeA'
    field_template = 'tab-b-c{}-{}-mo'
    file = 'SP0_00003.jpg'

    data = []

    for i in range(1, 17):
        for j in (1, 2, 3, 4, 6, 9, 12):
            field = field_template.format(i, j)
            image = Image.open(os.path.join(root, field, file))

            data.append([field, *image.size])

    data = pd.DataFrame(data, columns=['field', 'width', 'height'])

    print(data[['width', 'height']].value_counts())
    # >>>
    # width  height
    # 98     65        91
    # 121    72        21
    # dtype: int64


if __name__ == '__main__':
    main()
