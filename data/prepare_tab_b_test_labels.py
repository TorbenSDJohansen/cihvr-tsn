# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import os
import numpy as np


def main():
    indir = r'Y:\RegionH\Scripts\users\tsdj\storage\labels-root\210304-tab-b-cmd-tsdj-merge\tab_b_test_all_cells'
    outdir = r'Y:\RegionH\Scripts\data\storage\labels\tab-b-100x112-test-set\test'

    labels = {f: np.load(os.path.join(indir, f)) for f in os.listdir(indir)}
    labels_arr = np.concatenate([np.c_[v, np.repeat(k, len(v))] for k, v in labels.items()])
    unique = np.unique(np.concatenate(list(labels.values()))[:, 1])

    # [['SP4_19748.pdf.page-0.jpg' '11' 'tab-b-c16-2-mo.npy']] OK!
    # [['SPJ_2014-07-25_0351.PDF.page-0.jpg' '10' 'tab-b-c6-9-mo.npy'] OK!
    # [['SP2_30478.pdf.page-0.jpg' '23' 'tab-b-c16-6-mo.npy']] NOT OK!!!
    # Conclusion: Valid digits is 0-11, both inclusive.

    allowed = {'0=Mangler', '', *[str(i) for i in range(12)]}
    not_allowed = set(unique) - allowed
    print(f'Dropping all cases of {not_allowed}')

    for bad in not_allowed:
        sub = labels_arr[labels_arr[:, 1] == bad]

        if bad == 'x': # lots of these, cases we were not able to read
            continue

        if bad == 'b': # cannot use old bad CPD labels as new segmentation
            continue

        print(bad)
        print(sub)

    remap = {
        '': '0=Mangler',
        }
    assert set(remap.keys()).issubset(allowed)

    for k, v in labels.items():
        vsub = v.copy()
        vsub = vsub[np.isin(vsub[:, 1], list(allowed))] # NEED list instead of set
        vsub[:, 1] = [remap.get(x, x) for x in vsub[:, 1]]
        vsub[:, 0] = [x.split('.page-')[0] for x in vsub[:, 0]]
        vsub[:, 0] = [x.replace('.pdf', '.jpg').replace('.PDF', '.jpg') for x in vsub[:, 0]]

        for file in vsub[:, 0]:
            if not file.endswith('.jpg'):
                raise ValueError(file)

        np.save(os.path.join(outdir, k), vsub)


if __name__ == '__main__':
    main()
