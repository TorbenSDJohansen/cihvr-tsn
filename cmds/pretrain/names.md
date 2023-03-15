## Pre-training on PR
Danish names; see HANA for details on data.
Train one model for first names and other model for last names.

### Train

**For MH models**:
From DARE: Batch size 256 works with LR 0.5.
Now we scale this up to utilize most memory.
Note input size is 80x522 here and is 160x352 for DARE:

New batch size: 160 * 352 / (80 * 522) * 308 ~ 415 > 384 = 256 + 128
New LR: 0.5 * 384 / 256 * 2 = 1.5 (linear scaling on batch size)

Number of epochs (also warmup) follow from PR-1 model of DARE.

Last name model (MH):
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_keep_bad_cpd ^
--experiment last-mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr ^
--lr 1.5 ^
-b 384 ^
-j 8 ^
--input-size 3 80 522 ^
--epochs 90 ^
--warmup-epochs 5 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

Last name model (S2S); expect SeqAcc ~ 96.25%, see https://wandb.ai/tsdj/hana?workspace=user-tsdj:
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter s2s_last_name_keep_bad_cpd ^
--experiment last-s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr ^
--input-size 3 80 522 ^
--epochs 90 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

First name model (MH):
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_keep_bad_cpd ^
--experiment first-mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr ^
--lr 1.5 ^
-b 384 ^
-j 8 ^
--input-size 3 80 522 ^
--epochs 90 ^
--warmup-epochs 5 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

First name model (S2S)
```
python train.py ^
--formatter s2s_first_name_keep_bad_cpd ^
--experiment first-s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr ^
--input-size 3 80 522 ^
--epochs 90 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

### (Optional) Evaluate
Last:
```
python evaluate.py ^
--formatter last_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\pr\last ^
-b 2048 ^
--input-size 3 80 522 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\last\last.pth.tar
```

First:
```
python evaluate.py ^
--formatter first_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\pr\first ^
-b 2048 ^
--input-size 3 80 522 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\first\last.pth.tar
```