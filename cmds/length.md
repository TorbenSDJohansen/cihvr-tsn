# Length (at birth and 12 months) models
Merge cells to two image folders for all lengths (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields length-0-mo length-12-mo ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name length ^
--nb-pools 16
```

## Training
Base
```
python train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length ^
-b 512 ^
--input-size 3 93 198 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells length ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```