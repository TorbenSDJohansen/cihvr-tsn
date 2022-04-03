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
--formatter last_name_keep_bad_cpd ^
--experiment last ^
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

Last name model (EffNetV2-M - lower batch size and lr):
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_keep_bad_cpd ^
--experiment last-m ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr ^
--lr 0.375 ^
-b 96 ^
-j 8 ^
--input-size 3 80 522 ^
--epochs 90 ^
--warmup-epochs 5 ^
--data_dir "Z:\data_cropouts\Labels\HANA\HANA format" ^
--dataset HANA ^
--config ./cfgs/efficientnetv2_m.yaml ^
--log-wandb ^
--initial-log

```

First name model:
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_keep_bad_cpd ^
--experiment first ^
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

## Training

Notice larger image size, back to "default" epochs, smaller LR and batch size.
Also note we keep and predict bad cpd cases.

Last (base)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_keep_bad_cpd ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last ^
--lr 0.5 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 nurse-name-2 nurse-name-3 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

First (base)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_keep_bad_cpd ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first ^
--lr 0.5 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 nurse-name-2 nurse-name-3 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

Last (TL; search for LR; disable weight decay)
```
for %i in (1.0, 0.5, 0.25, 0.125, 0.0625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_keep_bad_cpd ^
--experiment tl-lr-%i ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last ^
--lr %i ^
--weight-decay 0 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 nurse-name-2 nurse-name-3 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\last\model_best.pth.tar ^
--log-wandb 

```

First (TL; search for LR; disable weight decay)
```
for %i in (1.0, 0.5, 0.25, 0.125, 0.0625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_keep_bad_cpd ^
--experiment tl-lr-%i ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first ^
--lr %i ^
--weight-decay 0 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 nurse-name-2 nurse-name-3 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\first\model_best.pth.tar ^
--log-wandb

```

## Evaluate

Last (base)
```
python evaluate.py ^
--formatter last_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\base ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 nurse-name-2 nurse-name-3 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

Last (TL, with post-match)
```
python evaluate.py ^
--formatter last_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\tl-lr-0.25 ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 nurse-name-2 nurse-name-3 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\tl-lr-0.25\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\tl-lr-0.25\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\ln-loose.pkl

```

First (base)
```
python evaluate.py ^
--formatter first_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\base ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 nurse-name-2 nurse-name-3 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

First (TL)
```
python evaluate.py ^
--formatter first_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\tl-lr-0.0625 ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 nurse-name-2 nurse-name-3 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\tl-lr-0.0625\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\tl-lr-0.0625\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\fn-loose.pkl
```

## Predict

Last (TL, only nurse-name-1, with post-match)
```
python predict.py ^
--formatter last_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\tl-lr-0.25 ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\tl-lr-0.25\last.pth.tar ^
--plots montage

python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\tl-lr-0.25\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\ln-loose.pkl
```

First (TL, only nurse-name-1, with post-match)
```
python predict.py ^
--formatter first_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\tl-lr-0.0625 ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\tl-lr-0.0625\last.pth.tar ^
--plots montage

python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\tl-lr-0.0625\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\fn-loose.pkl
```
