# APS_Thesis

Current Result

| ResNet50 | alpha = 0.1||        |      |      |      | alpha = 0.05 |          |          |       |       |      |
|----------|--------|--------|--------|------|------|------|----------|----------|----------|-------|-------|------|
|          |Coverage|Coverage|Coverage| Size | Size | Size | Coverage | Coverage | Coverage | Size  | Size  | Size |
| Datesets | APS    | RAPS   | SAPS   | APS  | RAPS | SAPS | APS      | RAPS     | SAPS     | APS   | RAPS  | SAPS |
| CIFAR10  | 89.61% | 89.69% | 90.09% | 1.41 | 1.14 | 1.02 | 94.69%   | 94.88%   | 95.01%   | 1.76  | 1.62  | 1.46 |
| CIFAR100 | 89.83% | 90.13% | 90.02% | 3.66 | 2.99 | 2.65 | 94.82%   | 94.85%   | 94.86%   | 6.40  | 6.21  | 5.83 |
| ImageNet | 89.90% | 90.02% | 90.01% | 14.89| 3.17 | 3.04 | 94.90%   | 95.04%   | 95.00%   | 31.76 | 10.08 | 7.72 |

| InceptionV3 | alpha = 0.1 |         |         |      |      |      | alpha = 0.05 |         |         |        |       |      |
|-------------|---------|---------|---------|------|------|------|---------|---------|---------|--------|-------|------|
|             | Coverage| Coverage| Coverage| Size | Size | Size | Coverage| Coverage| Coverage| Size   | Size  | Size |
| Datesets    | APS     | RAPS    | SAPS    | APS  | RAPS | SAPS | APS     | RAPS    | SAPS    | APS    | RAPS  | SAPS |
| CIFAR10     | 89.92%  | 89.89%  | 90.09%  | 1.81 | 1.51 | 1.18 | 94.95%  | 94.98%  | 94.99%  | 2.46   | 1.83  | 1.47 |
| CIFAR100    | 90.03%  | 90.04%  | 89.91%  | 3.44 | 3.08 | 3.06 | 94.96%  | 94.90%  | 95.05%  | 6.17   | 5.47  | 6.04 |
| ImageNet    | 89.96%  | 89.95%  | 90.05%  | 54.89| 3.14 | 2.55 | 94.98%  | 95.00%  | 95.00%  | 120.98 | 12.40 | 7.97 |

| ResNet18 | alpha = 0.1 |          |          |       |      |      | alpha = 0.05 |          |          |       |       |       |
|----------|-------------|----------|----------|-------|------|------|--------------|----------|----------|-------|-------|-------|
|          | Coverage    | Coverage | Coverage | Size  | Size | Size | Coverage     | Coverage | Coverage | Size  | Size  | Size  |
| Datesets | APS         | RAPS     | SAPS     | APS   | RAPS | SAPS | APS          | RAPS     | SAPS     | APS   | RAPS  | SAPS  |
| CIFAR10  | 89.47%      | 89.57%   | 89.88%   | 1.26  | 1.06 | 0.99 | 94.63%       | 94.57%   | 94.88%   | 1.55  | 1.47  | 1.27  |
| CIFAR100 | 90.17%      | 90.04%   | 90.14%   | 6.27  | 3.62 | 3.35 | 94.83%       | 94.85%   | 94.93%   | 11.13 | 8.27  | 7.86  |
| ImageNet | 89.89%      | 89.93%   | 89.94%   | 15.05 | 4.47 | 4.26 | 94.94%       | 94.96%   | 94.93%   | 31.15 | 11.18 | 11.09 | 

| ResNet34 | alpha = 0.1 |          |          |       |      |      | alpha = 0.05 |          |          |       |       |      |
|----------|-------------|----------|----------|-------|------|------|--------------|----------|----------|-------|-------|------|
|          | Coverage    | Coverage | Coverage | Size  | Size | Size | Coverage     | Coverage | Coverage | Size  | Size  | Size |
| Datesets | APS         | RAPS     | SAPS     | APS   | RAPS | SAPS | APS          | RAPS     | SAPS     | APS   | RAPS  | SAPS |
| CIFAR10  | 89.55%      | 89.46%   | 90.07%   | 1.23  | 1.18 | 0.99 | 94.70%       | 94.76%   | 94.93%   | 1.54  | 1.51  | 1.30 |
| CIFAR100 | 90.00%      | 89.94%   | 90.12%   | 3.67  | 3.51 | 2.85 | 94.90%       | 94.93%   | 94.85%   | 7.49  | 7.23  | 7.05 |
| ImageNet | 89.89%      | 89.93%   | 89.89%   | 14.56 | 4.47 | 3.07 | 94.96%       | 94.96%   | 94.95%   | 29.78 | 10.94 | 7.66 |

| VGG16    | alpha = 0.1 |          |          |       |      |      | alpha = 0.05 |          |          |       |       |       |
|----------|-------------|----------|----------|-------|------|------|--------------|----------|----------|-------|-------|-------|
|          | Coverage    | Coverage | Coverage | Size  | Size | Size | Coverage     | Coverage | Coverage | Size  | Size  | Size  |
| Datesets | APS         | RAPS     | SAPS     | APS   | RAPS | SAPS | APS          | RAPS     | SAPS     | APS   | RAPS  | SAPS  |
| CIFAR10  | 89.80%      | 89.90%   | 90.28%   | 1.13  | 1.02 | 0.99 | 94.88%       | 95.09%   | 95.20%   | 1.39  | 1.33  | 1.29  |
| CIFAR100 | 89.80%      | 89.82%   | 89.96%   | 4.86  | 4.72 | 4.40 | 94.91%       | 94.89%   | 94.78%   | 14.45 | 14.36 | 11.48 |
| ImageNet | 89.91%      | 89.92%   | 89.87%   | 11.70 | 3.51 | 2.85 | 94.88%       | 95.01%   | 94.96%   | 23.76 | 8.84  | 6.77  |

- Real Probability:

| CIFAR10-H   | alpha=0.1 |        |        |        |        |        | alpha=0.05 |       |        |         |        |        |
|-------------|---------|--------|--------|--------|--------|--------|--------|--------|--------|---------|--------|--------|
|             |avg. Prob|        |        |Frequency|       |        |avg.Prob|        |        |Frequency|        |        |
|             | APS     | RASP   | SAPS   | APS    | RAPS   | SAPS   | APS    | RAPS   | SAPS   | APS     | RAPS   | SAPS   |
| ResNet50    | 0.8693  | 0.8678 | 0.8714 | 57.79% | 55.80% | 55.48% | 0.9221 | 0.9230 | 0.9223 | 63.81%  | 63.24% | 62.12% |
| InceptionV3 | 0.8787  | 0.8754 | 0.8739 | 61.83% | 59.77% | 57.30% | 0.9339 | 0.9309 | 0.9253 | 71.31%  | 67.46% | 63.53% |

| ImageNet-Real | alpha=0.1 |        |        |        |        |        | alpha=0.05 |       |        |         |        |        |
|---------------|---------|--------|--------|--------|--------|--------|------------|--------|--------|---------|--------|--------|
|               |avg. Prob|        |        |Frequency|       |        | avg.Prob   |        |        |Frequency|        |        |
| ResNet50      | 0.7919  | 0.7496 | 0.7408 | 26.34% | 23.74% | 23.58% | 0.9221     | 0.9230 | 0.9223 | 63.81%  | 63.24% | 62.12% |
| InceptionV3   | 0.8093  | 0.7460 | 0.7257 | 29.90% | 23.44% | 21.61% | 0.9339     | 0.9309 | 0.9253 | 71.31%  | 67.46% | 63.53% |


Contribution Sheet:  
- Model: InceptionV3, ResNet18, ResNet34, ResNet50, VGG16
- Dataset: CIFAR10, CIFAR100, ImageNet-1k, CIFAR10-H
- For every combination between model and dataset, **all the algorithms(APS, RAPS and SAPS)** have been tested
- **CIFAR10**: all the models with two alpha 0.05 and 0.1
- **CIFAR100**: inception with four alphas(0.05, 0.1, 0.2, 0.3), the others with three alphas(0.05, 0.1, 0.2)
- **ImageNet**: all the models are tested with three alphas(0.05, 0.1, 0.2)
- **CIFAR10-H**: draw histograms and scatter-plots for all algorithms and repeat the experiments under two alphas(0.1, 0.05)
- **Synthetic Data**: Synthetic Datasets were categorized by num_classes(3, 5, 10). For all the datasets, 
draw histograms and scatter-plots for all algorithms and repeat the experiments under two alphas(0.1, 0.05)