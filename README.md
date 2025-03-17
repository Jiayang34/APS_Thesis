# APS_Thesis

Current Result

| ResNet50 | alpha = 0.1||        |      |      |      | alpha = 0.05 |          |          |       |      |      |
|----------|--------|--------|--------|------|------|------|--------------|----------|----------|-------|------|------|
|          |Coverage|Coverage|Coverage| Size | Size | Size | Coverage     | Coverage | Coverage | Size  | Size | Size |
| Datesets | APS    | RAPS   | SAPS   | APS  | RAPS | SAPS | APS          | RAPS     | SAPS     | APS   | RAPS | SAPS |
| CIFAR10  | 89.61% | 89.69% | 90.09% | 1.41 | 1.14 | 1.02 | 94.69%       | 94.88%   | 95.01%   | 1.76  | 1.62 | 1.46 |
| CIFAR100 | 89.83% | 90.13% | 90.02% | 3.66 | 2.99 | 2.65 | 94.82%       | 94.85%   | 94.86%   | 6.40  | 6.21 | 5.83 |
| ImageNet | 89.90% | 90.02% | 90.01% | 14.89| 3.17 | 3.04 | 89.90%       | 90.02%   | 90.01%   | 14.89 | 3.17 | 3.04 |

| InceptionV3 | alpha = 0.1 |         |         |      |      |      | alpha = 0.05 |          |          |       |      |      |
|-------------|---------|---------|---------|------|------|------|--------------|----------|----------|-------|------|------|
|             | Coverage| Coverage| Coverage| Size | Size | Size | Coverage     | Coverage | Coverage | Size  | Size | Size |
| Datesets    | APS     | RAPS    | SAPS    | APS  | RAPS | SAPS | APS          | RAPS     | SAPS     | APS   | RAPS | SAPS |
| CIFAR10     | 89.92%  | 89.89%  | 90.09%  | 1.81 | 1.51 | 1.18 | 94.95%       | 94.98%   | 94.99%   | 2.46  | 1.83 | 1.47 |
| CIFAR100    | 90.03%  | 90.04%  | 89.91%  | 3.44 | 3.08 | 3.06 | 94.96%       | 94.90%   | 95.08%   | 6.17  | 5.47 | 6.08 |
| ImageNet    | 89.96%  | 89.95%  | 90.05%  | 54.89| 3.14 | 2.55 | 89.90%       | 90.02%   | 90.01%   | 14.89 | 3.17 | 3.04 |

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
