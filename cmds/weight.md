# Weight models
Merge cells to two image folders for all weights (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\data\storage ^
--labels-subdir keep ^
--fields weight-0-mo weight-1-mo weight-2-mo weight-3-mo weight-4-mo weight-6-mo weight-9-mo weight-12-mo ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name weight ^
--nb-pools 8
```

**Note on image size**: Different for each weight-{0, 1, 2, 3, 4, 6, 9, 12}-mo (209x102, 262x72, 266x74, 259x75, 264x70, 258x80, 248x72, 255x72, respectively) even within Type A.
Use of 258x80 as that fits quite nicely (at least for all other than weight-0-mo, which is the most unique one, and of all the others it is the closest in terms of aspect ratio).

**NOTE**: Since multiple types, that resolution is not guaranteed for *all* examples -- only for that specific type (Type A).

## Training
MH
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter weight_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight ^
--lr 2.0 ^
-b 512 ^
--input-size 3 80 258 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells weight ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

S2S
```
python train.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight ^
-b 512 ^
--input-size 3 80 258 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells weight ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\weight\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\weight\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
MH
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\weight\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\mh\last.pth.tar ^
--plots montage ^
-b 2048 ^
--dataset image-datasets-joined ^
--dataset-cells weight-0-mo weight-1-mo weight-2-mo weight-3-mo weight-4-mo weight-6-mo weight-9-mo weight-12-mo
```

S2S
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\weight\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\weight\s2s\last.pth.tar ^
--plots montage ^
-b 2048 ^
--dataset image-datasets-joined ^
--dataset-cells weight-0-mo weight-1-mo weight-2-mo weight-3-mo weight-4-mo weight-6-mo weight-9-mo weight-12-mo
```
