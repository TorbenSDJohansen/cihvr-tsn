# Name (first and last) models

## Pre-training on PR
Danish names; see HANA for details on data.
Train model for first names and other model for last names.

### Train

From DARE: Batch size 256 works with LR 0.5.
Now we scale this up to utilize most memory.
Note input size is 80x522 here and is 160x352 for DARE:

New batch size: 160 * 352 / (80 * 522) * 308 ~ 415 > 384 = 256 + 128
New LR: 0.5 * 384 / 256 * 2 = 1.5 (linear scaling on batch size)

Number of epochs (also warmup) follow from PR-1 model of DARE.

Last name model:
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_long ^
--experiment pr-last-names ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names ^
--lr 1.5 ^
-b 384 ^
-j 8 ^
--input-size 3 80 522 ^
--epochs 90 ^
--warmup-epochs 5 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb

```

First name model:
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_long ^
--experiment pr-first-names ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names ^
--lr 1.5 ^
-b 384 ^
-j 8 ^
--input-size 3 80 522 ^
--epochs 90 ^
--warmup-epochs 5 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb

```

### Evaluate
Last:
```
python evaluate.py ^
--formatter last_name_long ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\pr-last-names ^
-b 2048 ^
--input-size 3 80 522 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr-last-names\last.pth.tar

```

First:
```
python evaluate.py ^
--formatter first_name_long ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\pr-first-names ^
-b 2048 ^
--input-size 3 80 522 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr-first-names\last.pth.tar

```

## Training

Notice larger image size, back to "default" epochs, smaller LR and batch size.

### Basline
Baseline model (from "scratch"; only "default" labels; no bad cpd).

Last:
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_long_cast_0 ^
--experiment last-names ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names ^
--lr 0.5 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets ^
--dataset nurse-names ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

First:

### TL from PR
**TODO**: Tune, esp. LR, but potentially also could be train for shorter, remove weight decay, etc.

Last:
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_long_cast_0 ^
--experiment last-names-tl ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names ^
--lr 0.5 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets ^
--dataset nurse-names ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr-last-names\last.pth.tar ^
--log-wandb 

```

Last (lower LR; disable weight decay):
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_long_cast_0 ^
--experiment last-names-tl_mod ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names ^
--lr 0.0625 ^
--weight-decay 0 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets ^
--dataset nurse-names ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr-last-names\last.pth.tar ^
--log-wandb 

```

## Evaluate

### Baseline

Last:
```
python evaluate.py ^
--formatter last_name_long_cast_0 ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last-names ^
-b 1024 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets ^
--dataset nurse-names ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last-names\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

**TODO**: For eval, only really nurse-name-1 is particularly interesting:
`--dataset-cells nurse-name-1`

