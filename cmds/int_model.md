# Model(s) for sequence of integers
Motivation: Following fields consist of sequence of integers with lengths 1-2/5
1. **Duration of any breastfeeding**: 1-2 digits
1. **Length at birth and one year**: 2 digits
1. **Preterm birth number of weeks**: 1-2 digits
1. **Table B visits information**: 1-2 digits
1. (Optional) **Weight at birth and at visits**: 4-5 digits

**Note**: Needs to be run after 
    [Duration of any breastfeeding](duration_any_bf.md),
    [Length at birth and one year](length.md),
    [Preterm birth number of weeks](preterm_weeks.md),
    [Table B visits information](tab_b.md),
    and [Weight at birth and at visits](cmds/weight.md),
as datasets otherwise won't exist.

**Note on image size**: 
Considerable heterogeneity in image sizes (both absolute and in terms of aspect ratio).
For individual models, following image sizes (aspect ratios) used:
1. **Duration of any breastfeeding**: 284x88 (~3.2)
1. **Length at birth and one year**: 297x109 (~2.7)
1. **Preterm birth number of weeks**: 193x100 (~1.9)
1. **Table B visits information**: 121x79 (1.5)
1. **Weight at birth and at visits**: 258x80 (~3.2)
Aim for middleground of aspect ratio of ~2.5 with image size 230x90
**NOTE**: Since multiple types, that resolution is not guaranteed for *all* examples -- only for that specific type (Type A).

## Training
S2S 5-digits (with weight)
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter s2s_five_digit_keep_bad_cpd ^
--experiment s2s-5d ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int ^
-b 512 ^
--input-size 3 90 230 ^
--data_dir Y:\RegionH\Scripts\users\tsdj\storage ^
--dataset image-datasets-train ^
--dataset-cells dura-any-breastfeed length preterm-birth-weeks tab-b weight ^
--config ./cfgs/deit3_b_s2s.yaml ^
--log-wandb ^
--initial-log
```

## Evaluate

### Duration any breastfeeding
S2S 5-digits
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\int-s2s-5d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
--dataset-cells dura-any-breastfeed ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

### Length
S2S 5-digits
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\int-s2s-5d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
--dataset-cells length ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

### Preterm number weeks
S2S 5-digits
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\int-s2s-5d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
--dataset-cells preterm-birth-weeks ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

### Table B
**~100x112 test set**: S2S 5-digits
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab-b\int-s2s-5d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
--dataset-cells tab-b-test ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

### Weight
S2S 5-digits
```
python evaluate.py ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\weight\int-s2s-5d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
--dataset-cells weight ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

## Predict
