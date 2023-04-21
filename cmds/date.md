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
**NOTE**: Currently TL is not used.

## Training
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
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
Current highest achieving model on test set is the MH model.
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\date\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\mh\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\date-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\date-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\date-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\date-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\date-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\date-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\date-12-mo ^
--plots montage
```