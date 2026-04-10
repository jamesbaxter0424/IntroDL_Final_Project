# 11785 Final Project: Floorplan Generation with ResPlan

[![arXiv](https://img.shields.io/badge/arXiv-2508.14006-b31b1b.svg)](https://arxiv.org/abs/2508.14006)

This repository contains our CMU 11-785 final project work built on top of the **ResPlan** dataset:

**[ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans](https://arxiv.org/abs/2508.14006)**  
*Mohamed Abouagour, Eleftherios Garyfallidis*

Original dataset repository:  
**[ResPlan GitHub Repo](https://github.com/m-agour/ResPlan)**

## Overview

We use the ResPlan floorplan dataset to train and evaluate graph-conditioned generative models for residential layout synthesis.

This repo currently includes:

- A **GraphGPS-based generator/discriminator pipeline**
- A **HouseGAN++ training notebook**
- Local and Kaggle-friendly dataset loading
- Checkpointing for `last`, `best`, and optional per-epoch saves
- W&B logging for metrics, previews, and best-model artifacts

## Repository Structure

- `Traingin_GraphGPS.ipynb`: main notebook for GraphGPS training, inference, and evaluation
- `Trainging_HouseGAN++.ipynb`: HouseGAN++ baseline / pretraining notebook
- `model/graphgps_models.py`: external GraphGPS model definitions used by the notebook
- `dataset/ResPlan.pkl`: local copy of the dataset
- `dataset/resplan_utils.py`: geometry, mask, and graph helper utilities
- `checkpoints/`: saved `.pth` checkpoints
- `outputs/`: previews, evaluation outputs, and W&B run files
- `requirements.txt`: Python dependencies

## Main Workflow

### 1. GraphGPS training

Open [Traingin_GraphGPS.ipynb](/c:/Users/james/Desktop/11785_final/Traingin_GraphGPS.ipynb).

This notebook handles:

- config setup
- dataset resolution for local / Kaggle / custom paths
- GraphGPS training
- checkpoint save / resume
- inference preview generation
- evaluation utilities such as FID / GED-related analysis

The GraphGPS model itself is defined in [model/graphgps_models.py](/c:/Users/james/Desktop/11785_final/model/graphgps_models.py), so you can iterate on architecture changes without duplicating notebook cells.

### 2. HouseGAN++ baseline

Open [Trainging_HouseGAN++.ipynb](/c:/Users/james/Desktop/11785_final/Trainging_HouseGAN++.ipynb) if you want to compare against the baseline pipeline.

## Installation

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If you prefer Conda, create an environment first and then install from `requirements.txt`.

## Dataset Setup

The repo currently expects the ResPlan PKL dataset to be available locally under:

```text
dataset/ResPlan.pkl
```

The notebooks also support multiple data-source modes through `cfg["data_source"]`:

- `local`
- `kaggle`
- `custom`

In the GraphGPS notebook, the relevant path settings are configured near the top of the notebook:

- `cfg["data_source"]`
- `cfg["local_dataset_root"]`
- `cfg["kaggle_dataset_root"]`
- `cfg["resplan_pkl_path"]`

## Checkpoints

GraphGPS checkpoints are saved to:

```text
checkpoints/<run_name>/last.pth
checkpoints/<run_name>/best.pth
checkpoints/<run_name>/config.yaml
checkpoints/<run_name>/graphgps_models.py
```

Optional periodic checkpoints are also saved when `save_every_n_epochs` is enabled.

```text
checkpoints/<run_name>/epoch_005.pth
```

Each checkpoint stores:

- epoch
- global step
- `model_name`
- `model_kwargs`
- generator / discriminator state dicts
- optimizer state dicts
- config snapshot
- metrics

This makes it easier to catch architecture mismatches when loading older checkpoints after model changes.

## Weights & Biases

The GraphGPS notebook supports W&B logging.

Current logging includes:

- training batch metrics
- epoch metrics
- preview images
- run summary fields

When a new **best checkpoint** is found, the notebook can also upload a W&B artifact containing:

- the best checkpoint file
- a YAML config snapshot
- the active GraphGPS model source file snapshot

This behavior is controlled by:

```python
cfg["use_wandb"] = True
cfg["upload_best_artifact_to_wandb"] = True
```

## Notes on Model Editing

If you want to change the GraphGPS architecture, edit:

- [model/graphgps_models.py](/c:/Users/james/Desktop/11785_final/model/graphgps_models.py)

The notebook imports:

```python
from model.graphgps_models import (
    MODEL_NAME,
    Generator,
    Discriminator,
    compute_gradient_penalty,
)
```

This keeps the training notebook easier to read and avoids copying model code across multiple notebooks.

## Citation

If you use **ResPlan** in research, please cite the original dataset paper:

```bibtex
@article{AbouagourGaryfallidis2025ResPlan,
  title   = {ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans},
  author  = {Abouagour, Mohamed and Garyfallidis, Eleftherios},
  journal = {arXiv preprint arXiv:2508.14006},
  year    = {2025},
  doi     = {10.48550/arXiv.2508.14006},
  url     = {https://arxiv.org/abs/2508.14006}
}
```

If you use or compare against the **HouseGAN++** baseline, please also cite:

Source: [House-GAN++ project page](https://ennauata.github.io/houseganpp/page.html)

```bibtex
@inproceedings{Nauata2021HouseGANPP,
  title     = {House-GAN++: Generative Adversarial Layout Refinement Network towards Intelligent Computational Agent for Professional Architects},
  author    = {Nauata, Nelson and Hosseini, Sepidehsadat and Chang, Kai-Hung and Chu, Hang and Cheng, Chin-Yi and Furukawa, Yasutaka},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
  url       = {https://ennauata.github.io/houseganpp/page.html}
}
```
