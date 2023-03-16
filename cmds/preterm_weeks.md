# Preterm number weeks models
Merge cells to two image folders (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields preterm-birth-weeks ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name preterm-birth-weeks ^
--nb-pools 16
```

## Training
MH
```
python train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks ^
-b 512 ^
--input-size 3 96 202 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells preterm-birth-weeks ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

S2S
```
python train.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks ^
-b 512 ^
--input-size 3 96 202 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells preterm-birth-weeks ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
MH
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm-wks\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks\mh\last.pth.tar ^
--plots montage ^
-b 2048 ^
--dataset image-datasets-joined ^
--dataset-cells preterm-birth-weeks
```

S2S
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm-wks\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\preterm-wks\s2s\last.pth.tar ^
--plots montage ^
-b 2048 ^
--dataset image-datasets-joined ^
--dataset-cells preterm-birth-weeks
```
