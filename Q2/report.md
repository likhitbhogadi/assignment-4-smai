# Question 2: Hyperparameter Tuning Report

## Overview

This report presents the results of a comprehensive hyperparameter sweep for a CNN model trained on the CIFAR-10 dataset. A total of **20 configurations** were explored using Weights & Biases (wandb) to systematically tune the model's hyperparameters.


Create sweep with ID: 3sdvzu8q  
Sweep URL: https://wandb.ai/likhitbhogadi-iiit-hyderabad/cifar10-colorization/sweeps/3sdvzu8q   
Random Sweep ID: 3sdvzu8q   
                                                              
## Experimental Setup

### Hyperparameter Search Space

The following hyperparameters were tuned across the sweep:

| Hyperparameter             | Values Explored          |
| -------------------------- | ------------------------ |
| **Learning Rate**          | {1e-4, 3e-4, 1e-3, 3e-3} |
| **Batch Size**             | {32, 64, 128}            |
| **Number of Filters (NF)** | {8, 16, 32}              |
| **Kernel Size**            | {3, 5}                   |
| **Optimizer**              | {SGD, Adam}              |

### Training Configuration

- **Number of Epochs**: 25 (for all runs)
- **Random Seed**: 42 (for reproducibility)
- **Dataset**: CIFAR-10
- **Sweep ID**: 3sdvzu8q

### Logged Metrics

For each run, the following were tracked:

- Training loss and validation loss curves (per epoch)
- Best validation loss achieved
- All hyperparameter configurations
- Model checkpoints (best model saved)

## Results Analysis

### Best Configuration

After evaluating all 20 configurations, the **best performing model** achieved the following:

| Metric                   | Value                       |
| ------------------------ | --------------------------- |
| **Run Name**             | lr0.0001_bs128_nf32_k5_Adam |
| **Best Validation Loss** | **1.4994**                  |
| **Final Training Loss**  | 1.5230                      |

#### Best Hyperparameters

```
Learning Rate:    0.0001
Batch Size:       128
Number of Filters: 32
Kernel Size:      5
Optimizer:        Adam
Epochs Trained:   25
```

### Top 5 Performing Configurations

| Rank | Run Name                    | Val Loss   | LR     | BS  | NF  | K   | Opt  |
| ---- | --------------------------- | ---------- | ------ | --- | --- | --- | ---- |
| 1    | lr0.0001_bs128_nf32_k5_Adam | **1.4994** | 0.0001 | 128 | 32  | 5   | Adam |
| 2    | lr0.0001_bs64_nf32_k5_Adam  | 1.5162     | 0.0001 | 64  | 32  | 5   | Adam |
| 3    | lr0.0003_bs32_nf16_k5_Adam  | 1.5478     | 0.0003 | 32  | 16  | 5   | Adam |
| 4    | lr0.001_bs32_nf32_k3_Adam   | 1.5785     | 0.001  | 32  | 32  | 3   | Adam |
| 5    | lr0.0001_bs32_nf16_k5_Adam  | 1.5895     | 0.0001 | 32  | 16  | 5   | Adam |

### Worst 5 Performing Configurations

| Rank | Run Name                   | Val Loss | LR     | BS  | NF  | K   | Opt  |
| ---- | -------------------------- | -------- | ------ | --- | --- | --- | ---- |
| 20   | lr0.0001_bs128_nf8_k3_SGD  | 2.8319   | 0.0001 | 128 | 8   | 3   | SGD  |
| 19   | lr0.0001_bs128_nf16_k5_SGD | 2.7996   | 0.0001 | 128 | 16  | 5   | SGD  |
| 18   | lr0.0001_bs64_nf8_k3_SGD   | 2.6394   | 0.0001 | 64  | 8   | 3   | SGD  |
| 17   | lr0.001_bs128_nf8_k3_SGD   | 2.1257   | 0.001  | 128 | 8   | 3   | SGD  |
| 16   | lr0.0001_bs128_nf8_k3_Adam | 1.9804   | 0.0001 | 128 | 8   | 3   | Adam |

## Key Insights

### 1. Optimizer Performance

**Adam significantly outperformed SGD:**

- **Best Adam configuration**: Validation loss = 1.4994
- **Best SGD configuration**: Validation loss = 1.7690 (lr0.003_bs128_nf32_k3_SGD)
- Adam configurations dominate the top-performing runs
- SGD struggled especially with low learning rates and small filter counts

### 2. Learning Rate Impact

**Lower learning rates (1e-4, 3e-4) generally performed better:**

- Optimal learning rate: **0.0001** (best overall performance)
- Learning rate 0.003 led to higher validation losses, indicating potential overfitting or instability
- Adam optimizer was more robust to learning rate variations than SGD

### 3. Batch Size Effect

**Larger batch sizes showed better performance with optimal configurations:**

- Best performance achieved with batch size **128**
- Batch size 64 and 32 also performed well when paired with appropriate hyperparameters
- Larger batches provided more stable gradients, especially beneficial for Adam optimizer

### 4. Number of Filters

**More filters generally improved performance:**

- **32 filters** consistently yielded better results
- 8 filters severely underperformed across all configurations
- The model benefits from higher representation capacity

### 5. Kernel Size

**Kernel size 5 showed slight advantage over 3:**

- Top configurations predominantly used kernel size 5
- Larger receptive field helped capture more spatial information
- The difference was more pronounced when combined with appropriate filter counts

### 6. Configuration Interactions

**Important synergies observed:**

- **Adam + Low LR + Large BS + Many Filters**: Best combination
- **SGD + Low LR**: Very poor performance (high losses ~2.8)
- **SGD + Higher LR (0.001-0.003)**: Improved but still inferior to Adam
- **Few filters (8) + Any configuration**: Consistently poor results

## Performance Distribution

### Validation Loss Statistics

```
Best Validation Loss:     1.4994
Worst Validation Loss:    2.8319
Mean Validation Loss:     1.8476
Median Validation Loss:   1.7740
Standard Deviation:       0.3976
```

### Configuration Count by Optimizer

- **Adam configurations**: 10 runs
- **SGD configurations**: 10 runs
- **Adam average val loss**: 1.6584
- **SGD average val loss**: 2.0368

## Recommendations

Based on the sweep results, the following recommendations are made:

1. **Use Adam optimizer** for this task - it consistently outperforms SGD
2. **Set learning rate to 1e-4** for stable, optimal convergence
3. **Use 32 filters** to provide sufficient model capacity
4. **Prefer kernel size 5** for better spatial feature extraction
5. **Use batch size 128** if memory permits, otherwise 64 is a good alternative
6. **Avoid very low filter counts (8)** as they severely limit model capacity
7. **If using SGD**, increase learning rate to at least 0.003 for acceptable performance

## Checkpoints and Artifacts

All run checkpoints have been saved in the `checkpoints_sweep/` directory with the following structure:

- Each run has a dedicated folder named by its hyperparameter configuration
- Best model weights saved as `best_model.pth` in respective folders
- The best overall model is stored in: `checkpoints_sweep/lr0.0001_bs128_nf32_k5_Adam/`

## WandB Run URL

**Sweep Dashboard**: [View on Weights & Biases - Sweep ID: 3sdvzu8q]

**Best Run**: lr0.0001_bs128_nf32_k5_Adam

_Note: The specific wandb run URLs can be accessed through the wandb dashboard using the sweep ID: `3sdvzu8q`_

## Conclusion

The hyperparameter sweep successfully identified an optimal configuration that achieves a validation loss of **1.4994**. The results clearly demonstrate that:

- **Adam optimizer** is superior for this CNN architecture and dataset
- **Lower learning rates** (1e-4) provide better generalization
- **Sufficient model capacity** (32 filters) is crucial for performance
- **Larger kernel sizes** (5x5) capture spatial features more effectively
- **Proper hyperparameter combinations** matter more than individual parameters

The systematic exploration of 20 configurations provided valuable insights into the model's behavior and sensitivities, enabling informed decisions for future model development and deployment.
