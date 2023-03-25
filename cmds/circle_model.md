# Breastfeeding 7-14 days old + Preterm birth model
Motivation: Both are circle number models:
1. **Breastfeeding 7-14 days old**: Circle 1, 2, or 3
1. **Preterm birth**: Circle 1 or 2

**Note**: Need to be run after [Breastfeeding at 7-14 days old](bf_7_days.md) and [Preterm birth](preterm.md) as datasets otherwise won't exist.

**Note on image size**:
Use of 537x117 and 193x100 for the individual models as that matches for Type A.
This is a large difference in terms of aspect ratio (~5 vs. ~2); aim for middleground at ~3.5 with 350x100.
**NOTE**: Since multiple types, that resolution is not guaranteed for *all* examples -- only for that specific type (Type A).

## Training
MH
```
python train.py ^
--formatter two_digit_keep_bad_cpd ^
--experiment mh ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\circle ^
--input-size 3 100 350 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells breastfeed-7-do preterm-birth ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```

S2S
```
python train.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--experiment s2s ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\circle ^
--input-size 3 100 350 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells breastfeed-7-do preterm-birth ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate


## Predict
