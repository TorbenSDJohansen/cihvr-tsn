# Dates

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

