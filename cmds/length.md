# Length (at birth and 12 months) models
Merge cells to two image folders for all lengths (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields length-0-mo length-12-mo ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name length ^
--nb-pools 16
```

## Training
MH
```
python train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length ^
-b 512 ^
--input-size 3 93 198 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells length ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

S2S
```
python train.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length ^
-b 512 ^
--input-size 3 93 198 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells length ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

S2S square 224x224
```
python train.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--experiment s2s-224x224 ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length ^
--input-size 3 224 224 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells length ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate
MH
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\mh\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\s2s ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\s2s\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\s2s\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

S2S square 224x224
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\s2s-224x224 ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\s2s-224x224\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\s2s-224x224\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
Base
```
python predict.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\length\mh ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\mh\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\length\mh\last.pth.tar ^
--plots montage ^
-b 2048 ^
--dataset image-datasets-joined ^
--dataset-cells length-0-mo length-12-mo
```