# Deep Learning Image Processing Template (DLIP)
This repository is the implementation of the BMT2022 Contribution [Annotation Efforts in Image Segmentation can be Reduced by Neural Network Bootstrapping](XXXX URL TO FOLLOW).

## Project Structure
Overview of this repository:
```
.
├── DLIP
│   ├── data #  Contains the defined datasets as PyTorch Lightning DataModules & Datasets.   
│   ├── experiments #  Contains experiment configurations as yaml files.
│   ├── models #  Contains the defined models as PyTorch Modules.    
│   ├── objectives #  Contains the defined objectives as PyTorch Modules.
│   ├── scripts #  Contains the training and inference scripts.
│   └── utils #  Contains utils functions, which can be used by all modules.
```

The training (`DLIP/scripts/train.py`) script is configured by the defined experiments (`DLIP/experiments`) and  utilize the defined datamodules (`DLIP/data`), models (`DLIP/models`) and objectives (`DLIP/objectives`).

## Install
### Prerequisite
- Python == 3.8.5
- Pip == 21.2.4
### Conda Environment
`conda create --name YOUR_ENV_NAME python=3.8.5`
### Pip Installation
1. Run `pip install -e .`
