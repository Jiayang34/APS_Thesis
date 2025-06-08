# Project Description

This repository stores the experiment progress and results of bachelor thesis  
"Evaluating Conformal Prediction in Image Classification".

# Environment Configuration
Python: `3.9.12`  
Cuda-toolkit: `11.8`  
Cudnn: `8.9`  
Pytorch: `2.5.0`  
Torchvision: `0.20.0`  
Torchaudio: `2.5.0`  
PyTorch-cuda: `11.8`  

# Repository Description

## src
The `src` directory contains all code and reusable functions, including:

- Implementation code for the APS, RAPS, and SAPS algorithms 
- Implementation code for the RAPS- and SAPS-Hyperparameter Optimization
- Implementation code for the conditional coverage tests
- Implementation code for generating Synthetic Datasets.
- Two versions of the Inception architecture, customized for CIFAR-10 and CIFAR-100 respectively  
- Code related to temperature scaling

## APS
The `APS` folder contains three subfolders: `CIFAR10`, `CIFAR100`, and `ImageNet-1K`, each corresponding to experiments   
conducted on the respective dataset (evaluating prediction set sizes). Each subfolder includes 5 `.ipynb` files, where   
each file represents an experiment using a specific model on the given dataset.Within each notebook, experiments are   
conducted using different values of alpha.  

## RAPS & SAPS
Similar to `APS`

## CIFAR10-H & ImageNet-Real 
`CIFAR10-H` and `ImageNet-Real` contains all the conditional coverage tests (including Histogram Tests and Scatterplot   
Tests) with five models and different alpha respectively.  

## Synthetic Data
`Synthetic Data` contains all the conditional coverage tests (including Histogram Tests and Scatterplot   
Tests) with three different K and alpha respectively.  

## Support
The directory `Support` contains all the operations that support experiments, including:

- Model accuracy tests
- Model training process
- Hyperparameter optimization
- Imagenet-Real and ImageNet reorganize