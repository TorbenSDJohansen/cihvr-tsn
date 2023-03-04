# Dates
Merge cells to two image folders for all dates (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name date ^
--nb-pools 8
```

## Pre-training on DARE
Dates from multiple sources; see DARE for details on data.
Using new formatter - but potentially worthwhile to also experiment with old.
Using the image size of dates in CIHVR, to match downstream task.
Epochs, warmup epochs, from DARE.
Note different formatter, need to allow year to be completely missing.

### Train
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter dates_keep_bad_cpd_allow_no_year ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\DARE ^
--lr 2.0 ^
-b 512 ^
--epochs 90 ^
--warmup-epochs 5 ^
-j 8 ^
--input-size 3 63 212 ^
--data_dir Z:\data_cropouts\Labels\DARE ^
--dataset cihvr-mod funeral-records swedish-records-death-dates ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

### (Optional) Evaluate
On all sets
```
python evaluate.py ^
--formatter dates_keep_bad_cpd_allow_no_year ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\DARE\base ^
-b 2048 ^
-j 8 ^
--input-size 3 63 212 ^
--data_dir Z:\data_cropouts\Labels\DARE ^
--dataset cihvr-mod funeral-records swedish-records-death-dates ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\DARE\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

Only on CIHVR
```
python evaluate.py ^
--formatter dates_keep_bad_cpd_allow_no_year ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\DARE\base-only-cihvr-mod ^
-b 2048 ^
-j 8 ^
--input-size 3 63 212 ^
--data_dir Z:\data_cropouts\Labels\DARE ^
--dataset cihvr-mod ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\DARE\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

Verify same when proper data_dir
```
python evaluate.py ^
--formatter dates_keep_bad_cpd_allow_no_year ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\DARE\base-only-true-cihvr ^
-b 2048 ^
-j 8 ^
--input-size 3 63 212 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\DARE\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

## Training

Base (old formatter, only to compare and verify new approach works)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter dates_keep_bad_cpd_old ^
--experiment old ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date ^
--lr 4.0 ^
-b 1024 ^
--input-size 3 63 212 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

Base (new formatter)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter dates_keep_bad_cpd ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date ^
--lr 4.0 ^
-b 1024 ^
--input-size 3 63 212 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

TL (new formatter)
```
for %i in (1.0, 0.5, 0.25, 0.125, 0.0625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter dates_keep_bad_cpd ^
--experiment tl-%i ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date ^
--lr %i ^
--weight-decay 0 ^
-b 512 ^
-j 8 ^
--input-size 3 63 212 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\DARE\base\last.pth.tar ^
--log-wandb

```

## Evaluate
Base (old formatter, only to compare and verify new approach works)
```
python evaluate.py ^
--formatter dates_keep_bad_cpd_old ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\old ^
-b 1024 ^
--input-size 3 63 212 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\old\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

Base (new formatter)
```
python evaluate.py ^
--formatter dates_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\base ^
-b 1024 ^
--input-size 3 63 212 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

## Predict

Base (old formatter, only to compare and verify new approach works)
```
python predict.py ^
--formatter dates_keep_bad_cpd_old ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\date\old ^
-b 1024 ^
--input-size 3 63 212 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\old\last.pth.tar

```

Base (new formatter)
```
python predict.py ^
--formatter dates_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\date\base ^
-b 1024 ^
--input-size 3 63 212 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\base\last.pth.tar

```
