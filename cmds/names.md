# Name (first and last) models
Merge cells to two image folders for all names (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\data\storage ^
--labels-subdir keep ^
--fields nurse-name-1 nurse-name-2 nurse-name-3 ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name nurse-name ^
--nb-pools 8
```

**Note on image size**: Use of 91x530 as that matches nurse-name-1 (primary target) for Type A.
nurse-name-{2, 3} resolution both 105x535, so proportions hardly different.
**NOTE**: Since multiple types, that resolution is not guaranteed for *all* examples -- only for that specific type (Type A).

## Pre-training
See [Nurse names (pretraining)](pretrain/names.md).

## Training

### Last name
MH w/ TL from HANA
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter last_name_keep_bad_cpd ^
--experiment mh-tl ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last ^
-b 128 ^
--input-size 3 91 530 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\last-mh\last-new-style-format.pth.tar ^
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
--input-size 3 91 530 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\last-mh\last-new-style-format.pth.tar ^
--log-wandb
```

S2S w/ TL from HANA
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter s2s_last_name_keep_bad_cpd ^
--experiment s2s-tl ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last ^
--input-size 3 91 530 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/deit3_b_s2s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\last-s2s\last.pth.tar ^
--tl-from-input-size 3 80 522 ^
--log-wandb ^
--initial-log
```

### First name
MH w/ TL from HANA
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter first_name_keep_bad_cpd ^
--experiment mh-tl ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first ^
-b 128 ^
--input-size 3 91 530 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\first-mh\last-new-style-format.pth.tar ^
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
--input-size 3 91 530 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/efficientnetv2_s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\first-mh\last-new-style-format.pth.tar ^
--log-wandb
```

S2S w/ TL from HANA
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter s2s_first_name_keep_bad_cpd ^
--experiment s2s-tl ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first ^
--input-size 3 91 530 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/deit3_b_s2s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\first-s2s\last.pth.tar ^
--tl-from-input-size 3 80 522 ^
--log-wandb ^
--initial-log
```

## Evaluate

### Last name
MH w/ TL from HANA
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\mh-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\mh-tl\last.pth.tar ^
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

S2S w/ TL from HANA
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

### First name
MH w/ TL from HANA
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\mh-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl\last.pth.tar ^
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

S2S w/ TL from HANA
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\s2s-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\s2s-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\s2s-tl\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict

### Last name
S2S w/ TL from HANA
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\s2s-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s-tl\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-1 Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-2 Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-3 ^
--plots montage
```

### First name
MH w/ TL from HANA
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\mh-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-1 Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-2 Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-3 ^
--plots montage
```