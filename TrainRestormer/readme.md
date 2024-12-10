# Restormer

This code is designed for training the Restoration Transformer from scratch using pairs of Clean and Turbulence Distorted images.

A PyTorch implementation of Restormer based on CVPR 2022 paper by [Hao Ren](https://github.com/leftthomas/Restormer).
[Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881).

![Network Architecture](result/structure.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch=1.10.2 torchvision cudatoolkit -c pytorch
```

## Dataset

Link to Download Dataset (No need to manually download dataset): [ASU_Turbulence_Clean_100k_dataset_v1]([https://](https://huggingface.co/datasets/riponcs/ASU_Turbulence_Clean_100k_dataset_v1)https://)

Run this command to autometically download and prepare the dataset:

```
python TrainRestormer/preapareDataset.py
```

```
data/
├── ASU_Turbulence_Clean_100k_dataset_v1/
│   ├── train/
│   │   ├── turb/
│   │   │   ├── image-1.png
│   │   │   └── ...
│   │   └── noturb/
│   │       ├── image-1.png
│   │       └── ...
│   └── test/
└── ASU_Turbulence_Clean_100k_dataset_v2/
    └── (same structure as v1)
```

### Train Model

```
python main.py --data_name ASU_Turbulence_Clean_100k_dataset_v1--seed 0
```

### Test Model

Testing
After training, follow these steps:
Copy the trained model to Turb-Seg-Res/PretrainedModel/restormer_ASUSim_trained.pth
Run the demo:

```
python Turb-Seg-Res/Demo.py
```

Note: Training is performed on one NVIDIA A100 GPU (80G) with default configuration and takes approximately 2 days.
