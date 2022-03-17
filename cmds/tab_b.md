# Table B
Many cells - define here.
```
set DATASET-CELLS-TAB-B=tab-b-c1-1-mo tab-b-c2-1-mo tab-b-c3-1-mo tab-b-c4-1-mo tab-b-c5-1-mo tab-b-c6-1-mo tab-b-c7-1-mo tab-b-c8-1-mo tab-b-c1-6-mo tab-b-c2-6-mo tab-b-c3-6-mo tab-b-c4-6-mo tab-b-c5-6-mo tab-b-c6-6-mo tab-b-c7-6-mo tab-b-c8-6-mo tab-b-c15-1-mo tab-b-c15-2-mo tab-b-c15-3-mo tab-b-c15-4-mo tab-b-c15-6-mo tab-b-c15-9-mo tab-b-c15-12-mo
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



## Predict

