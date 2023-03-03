# Weight models
Merge cells to two image folders for all weights (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields weight-0-mo weight-1-mo weight-2-mo weight-3-mo weight-4-mo weight-6-mo weight-9-mo weight-12-mo ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name weight ^
--nb-pools 8
```

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
--dataset image-datasets-train ^
--dataset-cells weight ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate
Base
```

```

## Predict
Base
```

```