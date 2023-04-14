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
    and [Weight at birth and at visits](weight.md),
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

S2S 5-digits force max 2 digits
```
python evaluate.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\int-s2s-5d-restrict-2d ^
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

S2S 5-digits force max 2 digits
```
python evaluate.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\int-s2s-5d-restrict-2d ^
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
**Train fields**: S2S 5-digits force max 2 digits
```
python evaluate.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab-b\int-s2s-5d-restrict-2d-train-fields ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
--dataset-cells tab-b ^
--plots montage cov-acc cer-acc ^
--eval-plots-omit-most-occ 3
```

**~100x112 test set**: S2S 5-digits force max 2 digits
```
python evaluate.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab-b\int-s2s-5d-restrict-2d-full-table ^
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

### Duration any breastfeeding
S2S 5-digits force max 2 digits
```
python predict.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\dabf\int-s2s-5d-restrict-2d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\dura-any-breastfeed ^
--plots montage
```

### Length
S2S 5-digits force max 2 digits
```
python predict.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\length\int-s2s-5d-restrict-2d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\length-0-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\length-12-mo ^
--plots montage
```

### Preterm number weeks
S2S 5-digits force max 2 digits
```
python predict.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm-wks\int-s2s-5d-restrict-2d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\preterm-birth-weeks ^
--plots montage
```

### Table B
S2S 5-digits force max 2 digits
```
python predict.py ^
--formatter s2s_two_digit_keep_bad_cpd ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\pred\tab-b\int-s2s-5d-post-add-len-tab-b-restrict-2d ^
--config Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d-post-add-len-tab-b\args.yaml ^
--checkpoint Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\int\s2s-5d-post-add-len-tab-b\last.pth.tar ^
-b 2048 ^
--predict-folders Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c1-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c1-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c1-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c1-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c1-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c1-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c1-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c10-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c10-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c10-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c10-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c10-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c10-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c10-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c11-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c11-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c11-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c11-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c11-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c11-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c11-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c12-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c12-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c12-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c12-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c12-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c12-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c12-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c13-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c13-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c13-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c13-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c13-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c13-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c13-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c14-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c14-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c14-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c14-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c14-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c14-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c14-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c15-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c15-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c15-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c15-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c15-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c15-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c15-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c16-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c16-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c16-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c16-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c16-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c16-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c16-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c2-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c2-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c2-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c2-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c2-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c2-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c2-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c3-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c3-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c3-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c3-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c3-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c3-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c3-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c4-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c4-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c4-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c4-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c4-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c4-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c4-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c5-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c5-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c5-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c5-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c5-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c5-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c5-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c6-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c6-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c6-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c6-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c6-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c6-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c6-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c7-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c7-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c7-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c7-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c7-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c7-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c7-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c8-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c8-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c8-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c8-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c8-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c8-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c8-9-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c9-1-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c9-12-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c9-2-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c9-3-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c9-4-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c9-6-mo Y:\RegionH\Scripts\data\storage\minipics\TypeA\tab-b-c9-9-mo ^
--plots montage
```
