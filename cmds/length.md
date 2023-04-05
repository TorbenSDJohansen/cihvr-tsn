# Length (at birth and 12 months) models
Merge cells to two image folders for all lengths (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\data\storage ^
--labels-subdir keep ^
--fields length-0-mo length-12-mo ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name length ^
--nb-pools 8
```

**Note on image size**: Use of 297x109 as length-0-mo is 297x109 and length-12-mo is 216x89 for Type A.
**NOTE**: Since multiple types, that resolution is not guaranteed for *all* examples -- only for that specific type (Type A).

## Training
MH
```
python train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length ^
-b 512 ^
--input-size 3 109 297 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells length ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

S2S
```
python train.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length ^
-b 512 ^
--input-size 3 109 297 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells length ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
While MH outperforms int-s2s-5d-restrict-2d, still go with int-s2s-5d-restrict-2d as MH has never seen empty cases, and thus is speculated to generalize poorer.
For that reason, see [Integer model](int_model.md) for prediction code.

## Preparing workspaces to add labels of empty cases
No examples of empty fields in labels.
Create workspace by selecting predictions of empty and then go through those.
Since too few predicted "0=Mangler", select remaining by choosing predictions with lowest certainty.
```
python data/labelling/empty.py ^
--fn-preds Z:\faellesmappe\tsdj\cihvr-timmsn\pred\length\int-s2s-5d-restrict-2d\preds.csv ^
--label-dir Y:\RegionH\Scripts\data\storage\labels ^
--outdir Y:\RegionH\Scripts\users\tsdj\storage\datasets\length-fields-empty ^
-n 1000
```

Post manual check, map to file useable format for creating/adding to labels
```
python data/labelling/empty_wsp_to_label.py ^
--wsp-dir Y:\RegionH\Scripts\users\tsdj\storage\datasets\length-fields-empty ^
--fn-out Y:\RegionH\Scripts\users\tsdj\storage\datasets\length-fields-empty\new-labels.csv
```