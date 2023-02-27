# Table B
Many cells - define here.
```
set DATASET-CELLS-TAB-B=tab-b-c1-1-mo tab-b-c1-2-mo tab-b-c1-3-mo tab-b-c1-4-mo tab-b-c1-6-mo tab-b-c1-9-mo tab-b-c1-12-mo tab-b-c2-1-mo tab-b-c2-2-mo tab-b-c2-3-mo tab-b-c2-4-mo tab-b-c2-6-mo tab-b-c2-9-mo tab-b-c2-12-mo tab-b-c3-1-mo tab-b-c3-2-mo tab-b-c3-3-mo tab-b-c3-4-mo tab-b-c3-6-mo tab-b-c3-9-mo tab-b-c3-12-mo tab-b-c4-1-mo tab-b-c4-2-mo tab-b-c4-3-mo tab-b-c4-4-mo tab-b-c4-6-mo tab-b-c4-9-mo tab-b-c4-12-mo tab-b-c5-1-mo tab-b-c5-2-mo tab-b-c5-3-mo tab-b-c5-4-mo tab-b-c5-6-mo tab-b-c5-9-mo tab-b-c5-12-mo tab-b-c6-1-mo tab-b-c6-2-mo tab-b-c6-3-mo tab-b-c6-4-mo tab-b-c6-6-mo tab-b-c6-9-mo tab-b-c6-12-mo tab-b-c7-1-mo tab-b-c7-2-mo tab-b-c7-3-mo tab-b-c7-4-mo tab-b-c7-6-mo tab-b-c7-9-mo tab-b-c7-12-mo tab-b-c8-1-mo tab-b-c8-2-mo tab-b-c8-3-mo tab-b-c8-4-mo tab-b-c8-6-mo tab-b-c8-9-mo tab-b-c8-12-mo tab-b-c9-1-mo tab-b-c9-2-mo tab-b-c9-3-mo tab-b-c9-4-mo tab-b-c9-6-mo tab-b-c9-9-mo tab-b-c9-12-mo tab-b-c10-1-mo tab-b-c10-2-mo tab-b-c10-3-mo tab-b-c10-4-mo tab-b-c10-6-mo tab-b-c10-9-mo tab-b-c10-12-mo tab-b-c11-1-mo tab-b-c11-2-mo tab-b-c11-3-mo tab-b-c11-4-mo tab-b-c11-6-mo tab-b-c11-9-mo tab-b-c11-12-mo tab-b-c12-1-mo tab-b-c12-2-mo tab-b-c12-3-mo tab-b-c12-4-mo tab-b-c12-6-mo tab-b-c12-9-mo tab-b-c12-12-mo tab-b-c13-1-mo tab-b-c13-2-mo tab-b-c13-3-mo tab-b-c13-4-mo tab-b-c13-6-mo tab-b-c13-9-mo tab-b-c13-12-mo tab-b-c14-1-mo tab-b-c14-2-mo tab-b-c14-3-mo tab-b-c14-4-mo tab-b-c14-6-mo tab-b-c14-9-mo tab-b-c14-12-mo tab-b-c15-1-mo tab-b-c15-2-mo tab-b-c15-3-mo tab-b-c15-4-mo tab-b-c15-6-mo tab-b-c15-9-mo tab-b-c15-12-mo tab-b-c16-1-mo tab-b-c16-2-mo tab-b-c16-3-mo tab-b-c16-4-mo tab-b-c16-6-mo tab-b-c16-9-mo tab-b-c16-12-mo

```

## Training

Base
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\tab_b ^
--lr 2.0 ^
-b 512 ^
--input-size 3 64 125 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells %DATASET-CELLS-TAB-B% ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

## Evaluate
**NOTE**: This now includes (practically) all fields to some degree

Base 
```
python evaluate.py ^
--formatter two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b\base ^
-b 512 ^
--input-size 3 64 125 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells %DATASET-CELLS-TAB-B% ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\tab_b\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

Base (on our own test set, with all 112 cells)
```
python evaluate.py ^
--formatter two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b\base-full-table ^
--lr 2.0 ^
-b 512 ^
--input-size 3 64 125 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--labels-subdir keep-tab-b-test ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\tab_b\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```


## Predict

