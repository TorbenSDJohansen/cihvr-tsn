# (Selective) manual labelling
Below are examples of creating workspaces from predictions to use those to add more labels.
These are "selective" in the sense of trying to oversample empty images for fields where no/few empty images with labels were available.

## Preterm
Few examples of empty fields in labels.
Create workspace by selecting predictions of empty and then go through those.
```
python data/labelling/create_wsp.py ^
--fn-preds Z:\faellesmappe\tsdj\cihvr-timmsn\pred\preterm\circle-mh\preds.csv ^
--label-dir Y:\RegionH\Scripts\data\storage\labels ^
--outdir Y:\RegionH\Scripts\users\tsdj\storage\datasets\preterm-fields-empty ^
-n 1000
```

Post manual check, map to file useable format for creating/adding to labels
```
python data/labelling/wsp_to_label.py ^
--wsp-dir Y:\RegionH\Scripts\users\tsdj\storage\datasets\preterm-fields-empty ^
--fn-out Y:\RegionH\Scripts\users\tsdj\storage\datasets\preterm-fields-empty\new-labels.csv
```

## Duration any breastfeeding
Few examples of empty fields in labels.
Create workspace by selecting predictions of empty and then go through those.
```
python data/labelling/create_wsp.py ^
--fn-preds Z:\faellesmappe\tsdj\cihvr-timmsn\pred\dabf\int-s2s-5d-restrict-2d\preds.csv ^
--label-dir Y:\RegionH\Scripts\data\storage\labels ^
--outdir Y:\RegionH\Scripts\users\tsdj\storage\datasets\dabf-fields-empty ^
-n 1000
```

Post manual check, map to file useable format for creating/adding to labels
```
python data/labelling/wsp_to_label.py ^
--wsp-dir Y:\RegionH\Scripts\users\tsdj\storage\datasets\dabf-fields-empty ^
--fn-out Y:\RegionH\Scripts\users\tsdj\storage\datasets\dabf-fields-empty\new-labels.csv
```

## Length
No examples of empty fields in labels.
Create workspace by selecting predictions of empty and then go through those.
Since too few predicted "0=Mangler", select remaining by choosing predictions with lowest certainty.
```
python data/labelling/create_wsp.py ^
--fn-preds Z:\faellesmappe\tsdj\cihvr-timmsn\pred\length\int-s2s-5d-restrict-2d\preds.csv ^
--label-dir Y:\RegionH\Scripts\data\storage\labels ^
--outdir Y:\RegionH\Scripts\users\tsdj\storage\datasets\length-fields-empty ^
-n 1000
```

Post manual check, map to file useable format for creating/adding to labels
```
python data/labelling/wsp_to_label.py ^
--wsp-dir Y:\RegionH\Scripts\users\tsdj\storage\datasets\length-fields-empty ^
--fn-out Y:\RegionH\Scripts\users\tsdj\storage\datasets\length-fields-empty\new-labels.csv
```

## Table B
Very few (and for most fields 0) examples of empty as labels.
Create workspace by selecting predictions of empty and then go through those.
```
python data/labelling/create_wsp.py ^
--fn-preds Z:\faellesmappe\tsdj\cihvr-timmsn\pred\tab-b\s2s\preds.csv ^
--label-dir Y:\RegionH\Scripts\data\storage\labels ^
--outdir Y:\RegionH\Scripts\users\tsdj\storage\datasets\tab-b-fields-empty ^
-n 100 ^
--package-size 1000
```

Post manual check, map to file useable format for creating/adding to labels
```
python data/labelling/wsp_to_label.py ^
--wsp-dir Y:\RegionH\Scripts\users\tsdj\storage\datasets\tab-b-fields-empty ^
--fn-out Y:\RegionH\Scripts\users\tsdj\storage\datasets\tab-b-fields-empty\new-labels.csv
```