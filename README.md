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
A number of scripts refer to specific full paths.
To replicate results on different server, some work to change this is needed.

### Data preparation
Make sure to run script in the order listed below.

To create dictionaries to map between long/short forms of images/journals use `python data/gen_deep_shallow_map.py`.

To create dictionary to map between EPI data dump variable names and SDU variable names use `python data/gen_map_lookup_df.py`.

To create .csv with number of images per journal use `python data/nb_pages_pr_journal.py`.

To prepare nurse name data (before new labels from @Malthe are added, see `cmds/names.md` for details) use `python data/prepare_nurse_name_data.py`.

To prepare new EPI data dump and harmonize with original EPI data dump use `python data/prepare_extra_data_lise.py`.

To prepare the @CMD & @TSDJ ~100x112 Table B test labels use `python data/prepare_tab_b_test_labels.py`.

To prepare train and test labels use `python data/gen_labels.py --dir Y:\RegionH\Scripts\data\storage\labels\keep`.

**Note**: For details on how to cluster journal pages, see [Applications of machine learning in tabular document digitisation](https://www.tandfonline.com/doi/abs/10.1080/01615440.2023.2164879).

### Segmentation
To create figures showcasing different page heights, use `python data/check_image_by_dob.py`.

**TODO**: Refer to new @CMD code once ready and merged to main.

To merge Type A and Type B segmentations (by copying B to A), use `python data/move_images.py --in-folder Y:\RegionH\Scripts\data\storage\minipics\TypeB --out-folder Y:\RegionH\Scripts\data\storage\minipics\TypeA --pools 20 --cells breastfeed-7-do date-0-mo date-1-mo date-12-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo district-1 district-2 district-3 dura-any-breastfeed length-0-mo length-12-mo meals-7-do moth-civ-status nb-abort nb-liveborn nb-stillborn nurse-name-1 nurse-name-2 nurse-name-3 PKU-7-do preterm-birth preterm-birth-weeks tab-b-c0-1-mo tab-b-c0-12-mo tab-b-c0-2-mo tab-b-c0-3-mo tab-b-c0-4-mo tab-b-c0-6-mo tab-b-c0-9-mo tab-b-c1-1-mo tab-b-c1-12-mo tab-b-c1-2-mo tab-b-c1-3-mo tab-b-c1-4-mo tab-b-c1-6-mo tab-b-c1-9-mo tab-b-c10-1-mo tab-b-c10-12-mo tab-b-c10-2-mo tab-b-c10-3-mo tab-b-c10-4-mo tab-b-c10-6-mo tab-b-c10-9-mo tab-b-c11-1-mo tab-b-c11-12-mo tab-b-c11-2-mo tab-b-c11-3-mo tab-b-c11-4-mo tab-b-c11-6-mo tab-b-c11-9-mo tab-b-c12-1-mo tab-b-c12-12-mo tab-b-c12-2-mo tab-b-c12-3-mo tab-b-c12-4-mo tab-b-c12-6-mo tab-b-c12-9-mo tab-b-c13-1-mo tab-b-c13-12-mo tab-b-c13-2-mo tab-b-c13-3-mo tab-b-c13-4-mo tab-b-c13-6-mo tab-b-c13-9-mo tab-b-c14-1-mo tab-b-c14-12-mo tab-b-c14-2-mo tab-b-c14-3-mo tab-b-c14-4-mo tab-b-c14-6-mo tab-b-c14-9-mo tab-b-c15-1-mo tab-b-c15-12-mo tab-b-c15-2-mo tab-b-c15-3-mo tab-b-c15-4-mo tab-b-c15-6-mo tab-b-c15-9-mo tab-b-c16-1-mo tab-b-c16-12-mo tab-b-c16-2-mo tab-b-c16-3-mo tab-b-c16-4-mo tab-b-c16-6-mo tab-b-c16-9-mo tab-b-c2-1-mo tab-b-c2-12-mo tab-b-c2-2-mo tab-b-c2-3-mo tab-b-c2-4-mo tab-b-c2-6-mo tab-b-c2-9-mo tab-b-c3-1-mo tab-b-c3-12-mo tab-b-c3-2-mo tab-b-c3-3-mo tab-b-c3-4-mo tab-b-c3-6-mo tab-b-c3-9-mo tab-b-c4-1-mo tab-b-c4-12-mo tab-b-c4-2-mo tab-b-c4-3-mo tab-b-c4-4-mo tab-b-c4-6-mo tab-b-c4-9-mo tab-b-c5-1-mo tab-b-c5-12-mo tab-b-c5-2-mo tab-b-c5-3-mo tab-b-c5-4-mo tab-b-c5-6-mo tab-b-c5-9-mo tab-b-c6-1-mo tab-b-c6-12-mo tab-b-c6-2-mo tab-b-c6-3-mo tab-b-c6-4-mo tab-b-c6-6-mo tab-b-c6-9-mo tab-b-c7-1-mo tab-b-c7-12-mo tab-b-c7-2-mo tab-b-c7-3-mo tab-b-c7-4-mo tab-b-c7-6-mo tab-b-c7-9-mo tab-b-c8-1-mo tab-b-c8-12-mo tab-b-c8-2-mo tab-b-c8-3-mo tab-b-c8-4-mo tab-b-c8-6-mo tab-b-c8-9-mo tab-b-c9-1-mo tab-b-c9-12-mo tab-b-c9-2-mo tab-b-c9-3-mo tab-b-c9-4-mo tab-b-c9-6-mo tab-b-c9-9-mo weight-0-mo weight-1-mo weight-12-mo weight-2-mo weight-3-mo weight-4-mo weight-6-mo weight-9-mo`

### Transcription
See individual markdowns under `./cmds/` for how to train, evaluate, and predict for all models.
This will also include details on any pre-training on other datasets, details on how to prepare the datasets used to train the models, and details on how to expand the datasets between rounds (e.g., for nurse names).
Pre-training and adding data between rounds happens only for some models.

**TODO**: List order to run in and hyperlink to .md

### Post-transcription
To produce table with transcription accuracies:
```
python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\bf7do\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b\base-full-table\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\weight\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\base\preds.csv ^
--cihvr-duplicate-drop ^
--out-dir ./
```

To format predictions to wide form:
```
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\bf7do\base\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\dabf\base\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\date\base\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\length\base\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\XXX\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\XXX\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm\base\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm-wks\base\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\tab_b\base\preds.csv --use-cihvr-name-if-available
python data/format_preds_cihvr.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\weight\base\preds.csv --use-cihvr-name-if-available
```

To prepare data for upload to DST use `python data/prepare_data_dst.py`.

## License

## Citing

- [ ] `timmsn`, with reference to `timm`
- [ ] Research papers based on `timmsn`
- [ ] Other papers used in, e.g., TL, such as HANA

## TODO

- [ ] Mask for bad CPD samples
- [ ] Optimal label construction, see `./data/gen_labels.py`
