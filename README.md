# Copenhagen Infanct Health Visitor Records (CIHVR)
Transcripted code related to the papers:
1. The long-run effects of longer follow-up: Evidence on the importance of childhood health interventions from a historical trial
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
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install imutils timm
```

After making sure all dependencies are installed, use the following code to install `timmsn`.
```
pip install path/to/timm-sequence-net
```

## License

## Citing

## TODO