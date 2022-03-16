# Weight models

## Training

Base
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter weight_keep_bad_cpd ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight ^
--lr 2.0 ^
-b 512 ^
--input-size 3 63 302 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells weight-0-mo weight-1-mo weight-2-mo weight-3-mo weight-4-mo weight-6-mo weight-9-mo weight-12-mo ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```


## Evaluate



## Predict

