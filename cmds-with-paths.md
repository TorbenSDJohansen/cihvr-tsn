# Preterm number of weeks accuracy with +/- 1 weeks
```
python scripts/preterm_weeks_error_rate.py --files Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\int-s2s-5d\preds.csv --fn-out W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2\preterm_nb_weeks_acc.tex
```

# Compare old to new transcriptions
```
python scripts/compare_uploads.py --fn-old Y:\RegionH\Scripts\users\tsdj\data_to_dst\210526_upload.csv --fn-new Y:\RegionH\Scripts\users\tsdj\data_to_dst\24-04-2023-09-59-22-to-dst.csv --fn-out W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2\change-cov-and-similarity.tex
```

# Balance table(s)
```
python scripts/balance_tables.py --fn-in Y:\RegionH\Scripts\users\tsdj\data_to_dst\210526_upload.csv --fn-out W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2\balance-tab-old.tex
python scripts/balance_tables.py --fn-in Y:\RegionH\Scripts\users\tsdj\data_to_dst\24-04-2023-09-59-22-to-dst.csv --fn-out W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2\balance-tab-new.tex
```

# Accuracy tables

For all fields, barring names; recall these otherwise gets merged to one, due to same field containing both first and last name.
Use own 100x112 Table B test set.
Make use of "meta fields" (groupings of, e.g., all weight fields).
```
python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\bf7do\circle-s2s\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\mh\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm\circle-s2s\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab-b\int-s2s-5d-full-table\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\weight\mh\preds.csv ^
--cihvr-duplicate-drop ^
--suffix meta-fields ^
--out-dir W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2
```

No longer merge fields to "meta fields".
```
python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\bf7do\circle-s2s\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\mh\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm\circle-s2s\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab-b\int-s2s-5d-full-table\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\weight\mh\preds.csv ^
--cihvr-duplicate-drop ^
--suffix individual-fields ^
--no-field-merge ^
--out-dir W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2
```

Names, first and last each their own command (and merging all three nurse name fields).
```
python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\s2s-tl\preds_matched.csv ^
--cihvr-duplicate-drop ^
--suffix names-last ^
--out-dir W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2

python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\mh-tl\preds_matched.csv ^
--cihvr-duplicate-drop ^
--suffix names-first ^
--out-dir W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2
```

Table B on test split of the domain we train on.
Make use of "meta fields" (groupings of, e.g., all weight fields).
```
python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab-b\int-s2s-5d-train-fields\preds.csv ^
--cihvr-duplicate-drop ^
--suffix tab-b-train-domain-meta-fields ^
--out-dir W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2
```

No longer merge fields to "meta fields".
```
python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab-b\int-s2s-5d-train-fields\preds.csv ^
--cihvr-duplicate-drop ^
--suffix tab-b-train-domain-individual-fields ^
--no-field-merge ^
--out-dir W:\BDADSharedData\tsdj\cihvr\transcr-report-tabs-and-figs\report-2
```