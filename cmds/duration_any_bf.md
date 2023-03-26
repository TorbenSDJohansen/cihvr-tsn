# Duration any breastfeeding (12 months) models
Merge cells to two image folders (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\data\storage ^
--labels-subdir keep ^
--fields dura-any-breastfeed ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name dura-any-breastfeed ^
--nb-pools 8
```

**Note on image size**: Use of 284x88 as that matches for Type A.
**NOTE**: Since multiple types, that resolution is not guaranteed for *all* examples -- only for that specific type (Type A).

## Training
MH
```
python train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf ^
-b 512 ^
--input-size 3 88 284 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells dura-any-breastfeed ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

S2S
```
python train.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf ^
-b 512 ^
--input-size 3 88 284 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells dura-any-breastfeed ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
MH
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\dabf\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf\mh\last.pth.tar ^
--plots montage ^
-b 2048 ^
--dataset image-datasets-joined ^
--dataset-cells dura-any-breastfeed
```

S2S
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\dabf\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\dabf\s2s\last.pth.tar ^
--plots montage ^
-b 2048 ^
--dataset image-datasets-joined ^
--dataset-cells dura-any-breastfeed
```
