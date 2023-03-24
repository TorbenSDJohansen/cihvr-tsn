# Dates
Merge cells to two image folders for all dates (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\data\storage ^
--labels-subdir keep ^
--fields date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name date ^
--nb-pools 8
```

**Note on image size**: Use of 181x67 as that matches for Type A for all date-{1, 2, 3, 4, 6, 9, 12}-mo.
**NOTE**: Since multiple types, that resolution is not guaranteed for *all* examples -- only for that specific type (Type A).

## Pre-training
See [Date of visits (pretraining)](pretrain/date.md).

## Training

MH (old formatter, only to compare and verify new approach works)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter dates_keep_bad_cpd_old ^
--experiment mh-old-formatter ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date ^
--lr 4.0 ^
-b 1024 ^
--input-size 3 67 181 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells date ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

MH
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter dates_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date ^
--lr 2.0 ^
-b 512 ^
--input-size 3 67 181 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells date ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

MH w/ TL
```
for %i in (1.0, 0.5, 0.25, 0.125, 0.0625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter dates_keep_bad_cpd ^
--experiment mh-tl-lr=%i ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date ^
--lr %i ^
--weight-decay 0 ^
-b 512 ^
-j 8 ^
--input-size 3 67 181 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells date ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\DARE\base\last.pth.tar ^
--log-wandb
```

S2S
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter s2s_dates_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date ^
-b 512 ^
--input-size 3 67 181 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells date ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate
MH (old formatter, only to compare and verify new approach works)
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\mh-old-formatter ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh-old-formatter\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh-old-formatter\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
MH
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\date\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\last.pth.tar ^
--plots montage ^
-b 2048 ^
--dataset image-datasets-joined ^
--dataset-cells date-1-mo date-2-mo date-3-mo date-4-mo date-6-mo date-9-mo date-12-mo
```
