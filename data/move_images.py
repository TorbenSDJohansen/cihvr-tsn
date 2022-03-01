# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:35:41 2022

@author: sa-tsdj
"""


import os
import shutil
import argparse
import multiprocessing


class MPCopier():
    def __init__(self, indir, outdir):
        self.indir = indir
        self.outdir = outdir

    def copy(self, file):
        outfile = os.path.join(self.outdir, file)
        infile = os.path.join(self.indir, file)
        shutil.copyfile(infile, outfile)


def _mp_copy(cells, inroot, outroot, nb_pools):
    for cell in cells:
        indir = os.path.join(inroot, cell)
        outdir = os.path.join(outroot, cell)

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=False)

        files = os.listdir(indir)
        existing_files = set(os.listdir(outdir))
        new_files = [x for x in files if x not in existing_files]
        print(f'Copying {len(new_files)} of {len(files)} files from "{indir}" to "{outdir}"!')

        mp_copier = MPCopier(indir, outdir)
        mp_copy = mp_copier.copy

        with multiprocessing.Pool(nb_pools) as pool:
            pool.map(mp_copy, new_files)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', type=str, nargs='+')
    parser.add_argument('--in-folder', type=str)
    parser.add_argument('--out-folder', type=str)
    parser.add_argument('--pools', type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    _mp_copy(args.cells, args.in_folder, args.out_folder, args.pools)


if __name__ == '__main__':
    main()
