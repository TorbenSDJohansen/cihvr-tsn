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
conda create -n cihvr numpy pandas pillow scikit-learn opencv matplotlib pyyaml
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

To prepare nurse name data (before new labels from @Malthe are added, see `cmds/names.md` for details) use `python data/prepare_nuse_name_data.py`. **Note**: Typo in name, "nuse" is meant to be "nurse".

To prepare new EPI data dump and harmonize with original EPI data dump use `python data/prepare_extra_data_lise.py`.

To prepare the @CMD & @TSDJ ~100x112 Table B test labels use `python data/prepare_tab_b_test_labels.py`.

To prepare train and test labels use `python data/gen_labels.py`.

**Note**: For details on how to cluster journal pages, see [Applications of machine learning in tabular document digitisation](https://www.tandfonline.com/doi/abs/10.1080/01615440.2023.2164879).

### Segmentation
To create figures showcasing different page heights, use `python data/check_image_by_dob.py`.

**TODO**: Refer to new @CMD code once ready and merged to main.

### Transcription
See individual markdowns under `./cmds/` for how to train, evaluate, and predict for all models.
This will also include details on any pre-training on other datasets, details on how to prepare the datasets used to train the models, and details on how to expand the datasets between rounds (e.g., for nurse names).
Pre-training and adding data between rounds happens only for some models.

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
