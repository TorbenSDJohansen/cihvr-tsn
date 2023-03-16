# Name (first and last) models
Merge cells to two image folders for all names (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields nurse-name-1 nurse-name-2 nurse-name-3 ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name nurse-name ^
--nb-pools 8
```

## Training
Notice larger image size, back to "default" epochs, smaller LR and batch size.
Also note we keep and predict bad cpd cases.

### Last name
MH
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last ^
--lr 0.5 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

MH (TL; search for LR; disable weight decay)
```
for %i in (1.0, 0.5, 0.25, 0.125, 0.0625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_keep_bad_cpd ^
--experiment mh-tl-lr=%i ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last ^
--lr %i ^
--weight-decay 0 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\last\last.pth.tar ^
--log-wandb
```

S2S. Note: 224 ** 2 / (95 * 650) ~ 0.8, decrease image size: (224 ^ 2 / (95 * 650)) ^ 0.5 * 95 ~ 85
```
python train.py ^
--formatter s2s_last_name_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last ^
--input-size 3 85 585 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

### First name
MH
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first ^
--lr 0.5 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

MH (TL; search for LR; disable weight decay)
```
for %i in (1.0, 0.5, 0.25, 0.125, 0.0625) DO python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_keep_bad_cpd ^
--experiment mh-tl-lr=%i ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first ^
--lr %i ^
--weight-decay 0 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\first\last.pth.tar ^
--log-wandb

```

S2S. Note: 224 ** 2 / (95 * 650) ~ 0.8, decrease image size: (224 ^ 2 / (95 * 650)) ^ 0.5 * 95 ~ 85
```
python train.py ^
--formatter s2s_first_name_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first ^
--input-size 3 85 585 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate

### Last name
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

MH (TL, with post-match)
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\mh-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh-tl-lr=XXX\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh-tl-lr=XXX\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\mh-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\ln-loose.pkl
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

### First name
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

MH (TL, with post-match)
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\mh-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl-lr=XXX\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl-lr=XXX\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\mh-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\fn-loose.pkl
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict

### Last name
MH (TL, only nurse-name-1, with post-match)
```
python predict.py ^
--formatter last_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\mh-tl ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh-tl-lr=XXX\last.pth.tar ^
--plots montage

python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\mh-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\ln-loose.pkl
```

### First name
MH (TL, only nurse-name-1, with post-match)
```
python predict.py ^
--formatter first_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\mh-tl ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl-lr=XXX\last.pth.tar ^
--plots montage

python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\mh-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\fn-loose.pkl
```

# Post round 1 revision
To create workspaces for more labelling:
```
python data\labelling\prepare_nurse_name_1.py --round 1 --task create-workspaces
```

To create labels from returned workspaces:
```
python data\labelling\prepare_nurse_name_1.py --round 1 --task create-labels
```

## Train

### Last name
MH (TL, use extra data)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_keep_bad_cpd ^
--experiment mh-tl-extra-data ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last ^
--lr XXX ^
--weight-decay 0 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\last\last.pth.tar ^
--log-wandb
```

### First name
MH (TL, use extra data)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_keep_bad_cpd ^
--experiment mh-tl-extra-data ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first ^
--lr XXX ^
--weight-decay 0 ^
-b 128 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\first\last.pth.tar ^
--log-wandb
```

## Evaluate

### Last name
MH (TL, use extra data)
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\mh-tl-extra-data ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh-tl-extra-data\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh-tl-extra-data\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

### First name
MH (TL, use extra data)
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\mh-tl-extra-data ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl-extra-data\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl-extra-data\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict/transcribe

### Last name
MH (TL, use extra data, only nurse-name-1)
```
python predict.py ^
--formatter last_name_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\mh-tl-extra-data ^
-b 2048 ^
--input-size 3 95 680 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells nurse-name-1 ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh-tl-extra-data\last.pth.tar ^
--plots montage
```
