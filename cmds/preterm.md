# Preterm models
Merge cells to two image folders (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\data\storage ^
--labels-subdir keep ^
--fields preterm-birth ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name preterm-birth ^
--nb-pools 8
```

**Note on image size**: Use of 249x107 as that matches for Type A.
**NOTE**: Since multiple types, that resolution is not guaranteed for *all* examples -- only for that specific type (Type A).

## Training
MH
```
python train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm ^
-b 512 ^
--input-size 3 107 249 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells preterm-birth ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

S2S
```
python train.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm ^
-b 512 ^
--input-size 3 107 249 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells preterm-birth ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
Current highest achieving model on test set is the circle-mh model (tie with MH).
For that reason, see [Circle model](circle_model.md) for prediction code.

## Preparing workspaces to add labels of empty cases
Few examples of empty fields in labels.
Create workspace by selecting predictions of empty and then go through those.
```
python data/labelling/empty.py ^
--fn-preds Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm\circle-mh\preds.csv ^
--label-dir Y:\RegionH\Scripts\data\storage\labels ^
--outdir Y:\RegionH\Scripts\users\tsdj\storage\datasets\preterm-fields-empty ^
-n 1000
```

Post manual check, map to file useable format for creating/adding to labels
```
python data/labelling/empty_wsp_to_label.py ^
--wsp-dir Y:\RegionH\Scripts\users\tsdj\storage\datasets\preterm-fields-empty ^
--fn-out Y:\RegionH\Scripts\users\tsdj\storage\datasets\preterm-fields-empty\new-labels.csv
```