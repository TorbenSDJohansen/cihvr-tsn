# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Theorize that labels "0" in EPI data dump for preterm birth weeks is in fact
wrong and should be "0=Mangler". Automatically change this when writing labels
for preterm birth weeks, but important to check this is actually the case.

Manually reviewing 333 examples (crashed after 333*), only 2 exceptions:
    (1) SP6_00881
    (2) SPJ_2014-04-30_0125

* Based on labels requested image file that did not exist.

"""


import os
import shutil

import numpy as np
import pandas as pd


def main():
    template_str = r'Y:\RegionH\Scripts\data\storage\labels\keep\{}\preterm-birth-weeks.npy'

    labels = np.concatenate([np.load(template_str.format(x), allow_pickle=True) for x in ('train', 'test')])
    labels = pd.DataFrame(labels, columns=['fname', 'label'])
    labels['path'] = r'Y:\RegionH\Scripts\data\storage\minipics\TypeA\preterm-birth-weeks\\' + labels['fname']

    # Select supposedly empty
    labels = labels[labels['label'] == '0=Mangler']

    nb_to_check: int = 500
    out_dir: str = './tmp-output/'

    np.random.seed(42)
    selected = np.random.choice(labels['path'], size=nb_to_check, replace=False)

    for file in selected:
        basename = os.path.basename(file)
        outname = os.path.join(out_dir, basename)
        shutil.copy(file, outname)


if __name__ == '__main__':
    main()
