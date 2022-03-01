# Name (first and last) models

## Pre-training on PR
Danish names; see HANA for details on data.
Train model for first names and other model for last names.

### Train

Last name model:
```
python -m torch.distributed.launch nproc_per_node=2 ^
--formatter last-name-long ^
--experiment pr-last-names ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\names ^
--input-size 3 80 522 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

### Evaluate
