# Table B + length + duration any breastfeeding
Motivation: All within two digit framework.
**NOTE**: Some potential for preterm-birth-weeks.

Many cells - define here.
```
set DATASET-CELLS-TAB-B-LEN-DABF=dura-any-breastfeed length-0-mo length-12-mo tab-b-c1-1-mo tab-b-c2-1-mo tab-b-c3-1-mo tab-b-c4-1-mo tab-b-c5-1-mo tab-b-c6-1-mo tab-b-c7-1-mo tab-b-c8-1-mo tab-b-c1-6-mo tab-b-c2-6-mo tab-b-c3-6-mo tab-b-c4-6-mo tab-b-c5-6-mo tab-b-c6-6-mo tab-b-c7-6-mo tab-b-c8-6-mo tab-b-c15-1-mo tab-b-c15-2-mo tab-b-c15-3-mo tab-b-c15-4-mo tab-b-c15-6-mo tab-b-c15-9-mo tab-b-c15-12-mo
```

**Note**: Not obvious which image size!!
1. Table B:         3 64 125    r = 0.512
1. Length:          3 93 198    r = 0.4696969696969697
1. Dur. any BF:     3 93 293    r = 0.3174061433447099

So r = 0.41 probably not bad. Further, to not scale anything down too much, let h = 93 -> w = 227

## Training

Base
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\tab_b_len_dabf ^
--lr 2.0 ^
-b 512 ^
-j 8 ^
--input-size 3 93 227 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells %DATASET-CELLS-TAB-B-LEN-DABF% ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log

```

## Evaluate
Base
```
python evaluate.py ^
--formatter two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b_len_dabf\base ^
-b 512 ^
-j 8 ^
--input-size 3 93 227 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--dataset-cells %DATASET-CELLS-TAB-B-LEN-DABF% ^
--labels-subdir keep ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\tab_b_len_dabf\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```

Base (on our own Table B test set, with all 112 cells)
```
python evaluate.py ^
--formatter two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b_len_dabf\base-full-table ^
-b 512 ^
-j 8 ^
--input-size 3 93 227 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-joined ^
--labels-subdir keep-tab-b-test ^
--config ./cfgs/efficientnetv2_s.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\tab_b_len_dabf\base\last.pth.tar ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3

```


## Predict

