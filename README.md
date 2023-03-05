# Copenhagen Infanct Health Visitor Records (CIHVR)
Transcripted code related to the papers:
1. Universal Investments in Toddler Health: Learning from a Large Government Trial
1. Cohort Profile: Copenhagen Infant Health Nurse Records (CIHNR) cohort
1. **nurse cluster project**

## Clone Repository and Prepare Environment
To get started, first clone the repository locally:
```
git clone https://github.com/TorbenSDJohansen/cihvr-tsn
```

Then prepare an environment (here using conda and the name `cihvr`):
```
conda create -n cihvr numpy pandas pillow scikit-learn opencv matplotlib pyyaml
conda activate cihvr
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install imutils timm
```

After making sure all dependencies are installed, use the following code to install `timmsn`.
```
pip install path/to/timm-sequence-net
```

## Replicate results
See individual markdowns under `./cmds/`.

To produce table with transcription accuracies:
```
python data/summ_preds.py ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\bf7do\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\dabf\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\tab_b\base-full-table\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\length\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\preterm-wks\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\weight\base\preds.csv ^
Z:\faellesmappe\tsdj\cihvr-timmsn\eval\date\base\preds.csv ^
--cihvr-duplicate-drop ^
--out-dir ./
```

## License

## Citing

- [ ] `timmsn`, with reference to `timm`
- [ ] Research papers based on `timmsn`
- [ ] Other papers used in, e.g., TL, such as HANA

## TODO

- [ ] Mask for bad CPD samples
- [ ] Optimal label construction, see `./data/gen_labels.py`
