# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:27:04 2022

@author: sa-tsdj
"""

import pandas as pd

p = pd.read_csv(r'Z:\faellesmappe\tsdj\cihvr-new\tab_b\eval-2021-05-24-09-58-03\eval-preds.csv')
p = pd.read_csv(r'Z:\faellesmappe\tsdj\cihvr-new\tab_b\eval-2021-05-24-10-02-39\eval-preds.csv')
p['cell'] = p['filename_full'].str.split('/').apply(lambda x: x[-2])

p = pd.read_csv(r'Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b\base\preds.csv')
p = pd.read_csv(r'Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b\base-full-table\preds.csv')
p['cell'] = p['filename_full'].str.split('\\').apply(lambda x: x[-1].split('/')[0])

p = pd.read_csv(r'Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b_len_dabf\base\preds.csv')
p = pd.read_csv(r'Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b_len_dabf\base-full-table\preds.csv')
p['cell'] = p['filename_full'].str.split('\\').apply(lambda x: x[-1].split('/')[0])

p['col'] = p['cell'].str.split('-').apply(lambda x: x[2])
p['c'] = p['pred'] == p['label']
p['c'].mean()
p.groupby('col')['c'].mean()
p.groupby('cell')['c'].mean()

p.loc[p['label'] != '0=Mangler', 'c'].mean()
p.loc[p['label'] != 'bad cpd', 'c'].mean()
