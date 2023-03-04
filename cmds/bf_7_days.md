# Breastfeeding status at 7-14 days old models
Merge cells to two image folders (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields breastfeed-7-do ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name breastfeed-7-do ^
--nb-pools 16
```

## Training
Base
```
python train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\bf7do ^
-b 512 ^
--input-size 3 96 202 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells breastfeed-7-do ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```