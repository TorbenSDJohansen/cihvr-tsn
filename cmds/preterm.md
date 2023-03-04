# Preterm models
Merge cells to two image folders (one for train and one for test):
```
python data\create_train_dataset.py ^
--dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-joined ^
--labels-subdir keep-restrict-share-bad-cpd ^
--fields preterm-birth ^
--out-dir Y:\RegionH\Scripts\users\tsdj\storage\image-datasets-train ^
--name preterm-birth ^
--nb-pools 16
```