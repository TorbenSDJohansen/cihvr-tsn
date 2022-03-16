# NAME-HERE

## Training

Need to change:
1. --formatter
1. --experiment
1. --outdir
1. --dataset-cells
1. --input-size
```
set FORMATTER=XYZ
set EXPERIMENT=XYZ
set OUTDIR-SUFFIX=XYZ
set DATASET-CELLS=XYZ
set INPUT-SIZE=3 X Y

python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter %FORMATTER% ^
--experiment %EXPERIMENT% ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\%OUTDIR-SUFFIX% ^
--lr 1.0 ^
--input-size %INPUT-SIZE% ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells %DATASET-CELLS% ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

## Evaluate



## Predict

