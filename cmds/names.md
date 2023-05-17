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

### First and last name
S2S w/ TL from HANA
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter s2s_first_and_last_name_keep_bad_cpd ^
--experiment s2s-tl ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first-and-last ^
--input-size 3 91 530 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells nurse-name ^
--config ./cfgs/deit3_b_s2s.yaml ^
--initial-checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\pr\first-and-last-s2s\last.pth.tar ^
--tl-from-input-size 3 80 522 ^
--log-wandb ^
--read-from-tar ^
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

MH w/ TL from HANA post w/ match
```
python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\mh-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\ln.pkl
```

S2S w/ TL from HANA
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\s2s-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s-tl\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S w/ TL from HANA post w/ match
```
python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\last\s2s-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\ln.pkl
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

MH w/ TL from HANA post w/ match
```
python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\mh-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\fn.pkl
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

S2S w/ TL from HANA post w/ match
```
python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first\s2s-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\fn.pkl
```

### First and last name
S2S w/ TL from HANA
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\names\first-and-last\s2s-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first-and-last\s2s-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first-and-last\s2s-tl\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict

### Last name
S2S w/ TL from HANA (not that there is much difference)
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\s2s-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\last\s2s-tl\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-1 Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-2 Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-3 ^
--plots montage
```

S2S w/ TL from HANA post w/ match
```
python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\last\s2s-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\ln.pkl
```

### First name
MH w/ TL from HANA (not that there is much difference)
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\mh-tl ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\names\first\mh-tl\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-1 Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-2 Y:\RegionH\Scripts\data\storage\minipics\TypeA\nurse-name-3 ^
--plots montage
```

MH w/ TL from HANA post w/ match
```
python match.py Z:\faellesmappe\tsdj\cihvr-timmsn\pred\names\first\mh-tl\preds.csv ^
--lex Y:\RegionH\Scripts\users\tsdj\storage\datasets\nurse-name-lex\fn.pkl
```