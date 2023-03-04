# Breastfeeding status at 7-14 days old models
Merge cells to two image folders (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields breastfeed-7-do ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name bresatfeed-7-do ^
--nb-pools 16
```