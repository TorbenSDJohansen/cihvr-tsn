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

### Evaluate
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