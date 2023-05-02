# Copenhagen Infanct Health Visitor Records (CIHVR)
Transcription code related to the papers:
1. Universal Investments in Toddler Health: Learning from a Large Government Trial
1. Cohort Profile: Copenhagen Infant Health Nurse Records (CIHNR) cohort
1. **Nurse cluster project**

## Clone Repository and Prepare Environment
To get started, first clone the repository locally:
```
git clone https://github.com/TorbenSDJohansen/cihvr-tsn
```

Then prepare an environment (here using conda and the name `cihvr`):
```
conda create -n cihvr numpy pandas pillow scikit-learn opencv matplotlib pyyaml xlrd openpyxl
conda activate cihvr
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install imutils timm
```

After making sure all dependencies are installed, use the following code to install `timmsn`.
```
pip install path/to/timm-sequence-net
```

## Replicate results
**Note**:
A number of scripts refer to specific full paths.
To replicate results on different server, some work to change this is needed.

### Data preparation
Make sure to run script in the order listed below.

To create dictionaries to map between long/short forms of images/journals use `python data/gen_deep_shallow_map.py`.

To create dictionary to map between EPI data dump variable names and SDU variable names use `python data/gen_map_lookup_df.py`.

To create .csv with number of images per journal use `python data/nb_pages_pr_journal.py`.

To prepare nurse name data (before new labels from @Malthe are added, see `data/labelling/prepare_nurse_name_1.py` and `data/gen_labels.py` for details) use `python data/prepare_nurse_name_data.py`.

To prepare new EPI data dump and harmonize with original EPI data dump use `python data/prepare_extra_data_lise.py`.

To prepare the @CMD & @TSDJ ~100x112 Table B test labels use `python data/prepare_tab_b_test_labels.py`.

To prepare train and test labels use `python data/gen_labels.py --dir Y:\RegionH\Scripts\data\storage\labels\keep`.

**Note**: For details on how to cluster journal pages, see [Applications of machine learning in tabular document digitisation](https://www.tandfonline.com/doi/abs/10.1080/01615440.2023.2164879).

### Segmentation
To create figures showcasing different page heights, use `python data/check_image_by_dob.py`.

**TODO**: Refer to new @CMD code once ready and merged to main.

To merge Type A and Type B segmentations (by copying B to A), use
```
python data/move_images.py --in-folder Y:\RegionH\Scripts\data\storage\minipics\TypeB --out-folder Y:\RegionH\Scripts\data\storage\minipics\TypeA --pools 20 --cells breastfeed-7-do date-0-mo date-1-mo date-12-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo district-1 district-2 district-3 dura-any-breastfeed length-0-mo length-12-mo meals-7-do moth-civ-status nb-abort nb-liveborn nb-stillborn nurse-name-1 nurse-name-2 nurse-name-3 PKU-7-do preterm-birth preterm-birth-weeks tab-b-c0-1-mo tab-b-c0-12-mo tab-b-c0-2-mo tab-b-c0-3-mo tab-b-c0-4-mo tab-b-c0-6-mo tab-b-c0-9-mo tab-b-c1-1-mo tab-b-c1-12-mo tab-b-c1-2-mo tab-b-c1-3-mo tab-b-c1-4-mo tab-b-c1-6-mo tab-b-c1-9-mo tab-b-c10-1-mo tab-b-c10-12-mo tab-b-c10-2-mo tab-b-c10-3-mo tab-b-c10-4-mo tab-b-c10-6-mo tab-b-c10-9-mo tab-b-c11-1-mo tab-b-c11-12-mo tab-b-c11-2-mo tab-b-c11-3-mo tab-b-c11-4-mo tab-b-c11-6-mo tab-b-c11-9-mo tab-b-c12-1-mo tab-b-c12-12-mo tab-b-c12-2-mo tab-b-c12-3-mo tab-b-c12-4-mo tab-b-c12-6-mo tab-b-c12-9-mo tab-b-c13-1-mo tab-b-c13-12-mo tab-b-c13-2-mo tab-b-c13-3-mo tab-b-c13-4-mo tab-b-c13-6-mo tab-b-c13-9-mo tab-b-c14-1-mo tab-b-c14-12-mo tab-b-c14-2-mo tab-b-c14-3-mo tab-b-c14-4-mo tab-b-c14-6-mo tab-b-c14-9-mo tab-b-c15-1-mo tab-b-c15-12-mo tab-b-c15-2-mo tab-b-c15-3-mo tab-b-c15-4-mo tab-b-c15-6-mo tab-b-c15-9-mo tab-b-c16-1-mo tab-b-c16-12-mo tab-b-c16-2-mo tab-b-c16-3-mo tab-b-c16-4-mo tab-b-c16-6-mo tab-b-c16-9-mo tab-b-c2-1-mo tab-b-c2-12-mo tab-b-c2-2-mo tab-b-c2-3-mo tab-b-c2-4-mo tab-b-c2-6-mo tab-b-c2-9-mo tab-b-c3-1-mo tab-b-c3-12-mo tab-b-c3-2-mo tab-b-c3-3-mo tab-b-c3-4-mo tab-b-c3-6-mo tab-b-c3-9-mo tab-b-c4-1-mo tab-b-c4-12-mo tab-b-c4-2-mo tab-b-c4-3-mo tab-b-c4-4-mo tab-b-c4-6-mo tab-b-c4-9-mo tab-b-c5-1-mo tab-b-c5-12-mo tab-b-c5-2-mo tab-b-c5-3-mo tab-b-c5-4-mo tab-b-c5-6-mo tab-b-c5-9-mo tab-b-c6-1-mo tab-b-c6-12-mo tab-b-c6-2-mo tab-b-c6-3-mo tab-b-c6-4-mo tab-b-c6-6-mo tab-b-c6-9-mo tab-b-c7-1-mo tab-b-c7-12-mo tab-b-c7-2-mo tab-b-c7-3-mo tab-b-c7-4-mo tab-b-c7-6-mo tab-b-c7-9-mo tab-b-c8-1-mo tab-b-c8-12-mo tab-b-c8-2-mo tab-b-c8-3-mo tab-b-c8-4-mo tab-b-c8-6-mo tab-b-c8-9-mo tab-b-c9-1-mo tab-b-c9-12-mo tab-b-c9-2-mo tab-b-c9-3-mo tab-b-c9-4-mo tab-b-c9-6-mo tab-b-c9-9-mo weight-0-mo weight-1-mo weight-12-mo weight-2-mo weight-3-mo weight-4-mo weight-6-mo weight-9-mo
```

### Transcription
See individual markdowns under `./cmds/` for all details on how to train, evaluate, and predict for all models.
This will also include details on any pre-training on other datasets and details on how to prepare the datasets used to train and evaluate the models.
Pre-training happens only for some models.

For full rundown, refer to markdowns in following order (some parts need to run before others, though the specific order listed below is not the only feasible order):
1. [Breastfeeding at 7-14 days old](cmds/bf_7_days.md)
1. [Date of visits](cmds/date.md)
1. [Duration of any breastfeeding](cmds/duration_any_bf.md)
1. [Length at birth and one year](cmds/length.md)
1. [Nurse names](cmds/names.md)
1. [Preterm birth](cmds/preterm.md)
1. [Preterm birth number of weeks](cmds/preterm_weeks.md)
1. [Table B visits information](cmds/tab_b.md)
1. [Weight at birth and at visits](cmds/weight.md)
1. [Circle number model](cmds/circle_model.md)
1. [Seq. of integers model](cmds/int_model.md)

### Post-transcription
To produce table with transcription accuracies:
```
python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\bf7do\circle-s2s\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\mh\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\s2s-tl\preds_matched.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\mh-tl\preds_matched.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm\circle-s2s\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab-b\int-s2s-5d\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\weight\mh\preds.csv ^
--cihvr-duplicate-drop ^
--out-dir path/to/out/
```
**FIXME** should we split this, currently we "merge" first and last name to one field in table as same name if field(s)

**Note**: Performance on [Preterm birth number of weeks](cmds/preterm_weeks.md) appears very poor.
However, this is in large part due to inconsistent labelling:
Images often contains a range, such as "5-6", whereas the label only contains 1 number and not a range, even if present on the image.
Further, it is not consistent whether the first or the second number in a range was used as the label.
To try to get a better measure of performance, calculate accuracy when allowing a difference of 1 between label and transcription, see `python scripts/preterm_weeks_error_rate.py --files Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\int-s2s-5d\preds.csv --fn-out path/to/output.tex`.

To format predictions to wide form:
```
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\bf7do\circle-s2s\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\dabf\int-s2s-5d\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\date\mh\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\length\int-s2s-5d\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\s2s-tl\preds_matched.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\mh-tl\preds_matched.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm\circle-s2s\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm-wks\int-s2s-5d\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\tab-b\int-s2s-5d\preds.csv
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\weight\mh\preds.csv --use-cihvr-name-if-available
```

To prepare data for upload to DST use `python data/prepare_data_dst.py`.

To compare to older upload use `python scripts/compare_uploads.py --fn-old path/to/old.csv --fn-new path/to/new --fn-out path/to/out.tex`.

To create balance table (with respect to born 1-3) use `python scripts/balance_tables.py --fn-in path/to/transcriptions.csv --fn-out path/to/out.tex`.

## License
Our code is licensed under Apache 2.0 (see [LICENSE](LICENSE)).

## Citing

- [ ] `timmsn`, with reference to `timm`
- [ ] Research papers based on `timmsn`
- [ ] Other papers used in, e.g., TL, such as HANA
