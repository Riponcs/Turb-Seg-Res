<div align="center">

# 【CVPR'2024🔥】Turb-Seg-Res: A Segment-then-Restore Pipeline for Dynamic Videos with Atmospheric Turbulence

</div>

## Useful Links
| Links | Description | 
|:-----: |:-----: |
| [![Website Demo](https://img.shields.io/badge/TurbSegRes-Website-blue)](https://riponcs.github.io/TurbSegRes/) | Official project page with detailed information | 
| [![GitHub](https://img.shields.io/badge/TurbSegRes-GitHub-blue)](https://github.com/Riponcs/Turb-Seg-Res) | Link to the GitHub repository |
| [![Paper](https://img.shields.io/badge/Paper-arXiv-green)](https://arxiv.org/abs/2404.13605) | Link to the CVPR 2024 paper |
| [![QuickTurbSim](https://img.shields.io/badge/QuickTurbSim-GitHub-blue)](https://github.com/Riponcs/QuickTurbSim) | Repository for simulating atmospheric turbulence effects |
| [![DOST Dataset](https://img.shields.io/badge/Dataset-DOST-orange)](https://turbulence-research.github.io/) | Dataset used in the project |

## Setup and Run
```sh
git clone https://github.com/Riponcs/Turb-Seg-Res.git
cd Turb-Seg-Res
pip install -r requirements.txt
python Demo.py
```
## Contributions
- **High Focal Length Video Stabilization:** Stabilizes videos captured by high focal length cameras, which are highly sensitive to vibrations.
- **Turbulence Video Simulation:** Introduces a novel tilt-and-blur video simulator based on simplex noise for generating plausible turbulence effects with temporal coherence.
- **Unsupervised Motion Segmentation:** Efficiently segments dynamic scenes affected by atmospheric turbulence, distinguishing between static and dynamic components.

## Usage (Demo.py)
The main script for running the demo is `Demo.py`. It processes a set of input images, applies stabilization, and saves the output images.

```sh
python Demo.py
```

### Configuration
The following configuration settings can be adjusted in `Demo.py`:

- `doStabilize`: Enable or disable image stabilization.
- `ProcessNumberOfFrames`: Number of frames to process from the input images.
- `resizeFactor`: Factor to resize images.
- `MaxStb`: Maximum allowed pixel shift for image stabilization.
- `path`: Path to input images.
- `savePath`: Path to save output images.

## Additional Resources
- [![QuickTurbSim](https://img.shields.io/badge/QuickTurbSim-GitHub-blue)](https://github.com/Riponcs/QuickTurbSim): A repository for simulating atmospheric turbulence effects on images using 3D simplex noise and Gaussian blur.

## Train the Model:
- To train the Restormer model, you can use this code:[Restormer Implementation](https://github.com/leftthomas/Restormer), and for Dataset Generation use the [Turbulence Simulator Code](https://github.com/Riponcs/QuickTurbSim) on [MIT Place Dataset](https://www.kaggle.com/datasets/nickj26/places2-mit-dataset). I'll update the training code and dataset to make it simple by the end of November.

## Citation
If you find this work useful, please cite our CVPR 2024 paper:
```bibtex
@article{saha2024turb,
    title     = {Turb-Seg-Res: A Segment-then-Restore Pipeline for Dynamic Videos with Atmospheric Turbulence},
    author    = {Saha, Ripon Kumar and Qin, Dehao and Li, Nianyi and Ye, Jinwei and Jayasuriya, Suren},
    booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    year      = {2024},
}
