## Pre-training on DARE
Dates from multiple sources; see [DARE Database](https://www.kaggle.com/datasets/sdusimonwittrock/dare-database) for details on data.
Using new formatter - but potentially worthwhile to also experiment with old.
Using the image size of dates in CIHVR, to match downstream task.
Epochs, warmup epochs, from DARE.
Note different formatter, need to allow year to be completely missing.

### Train
```
python -m torch.distributed.launch --nproc_per_node=2 train.py ^
--formatter dates_keep_bad_cpd_allow_no_year ^
--experiment base ^
--output Z:\faellesmappe\tsdj\cihvr-timmsn\experiments\date\DARE ^
--lr 2.0 ^
-b 512 ^
--epochs 90 ^
--warmup-epochs 5 ^
-j 8 ^
--input-size 3 63 212 ^
--data_dir Z:\data_cropouts\Labels\DARE ^
--dataset cihvr-mod funeral-records swedish-records-death-dates ^
--config ./cfgs/efficientnetv2_s.yaml ^
--log-wandb ^
--initial-log
```
