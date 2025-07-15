# GraphFW.runners

This module contains runner classes that handle the training, validation, and testing loops for graph classification experiments with advanced features like early stopping, model checkpointing, and learning rate scheduling.

## Contents
- `base_runner.py`: Base class for all runners with core training/validation logic.
- `kfold_runner.py`: K-Fold cross-validation runner with per-fold model management.
- `split_runner.py`: Train/validation/test split runner for standard evaluation.
- `utils/`: Progress bar and other runner utilities.
- `__init__.py`: Module initialization.

## Features

### Training Management
- **Early Stopping**: Configurable patience and improvement thresholds
- **Model Checkpointing**: Automatic saving of best models based on validation metrics
- **Learning Rate Scheduling**: Support for PyTorch LR schedulers
- **Progress Tracking**: Real-time training progress with loss and accuracy monitoring

### Cross-Validation Support
- **K-Fold CV**: Independent model training for each fold
- **Stratified Splitting**: Maintains class distribution across folds
- **Aggregate Statistics**: Mean and variance computation across folds

## Configuration

### Runner Configuration
```python
runner = dict(
    type='KFoldRunner',         # or 'SplitRunner'
    n_splits=10,                # Number of folds (K-Fold only)
    val_interval=1,             # Validation frequency (epochs)
    train_epochs='inf',         # Maximum epochs (early stopping controls actual training)
    log_interval=1,             # Logging frequency (iterations)
    metric='val_acc',           # Metric to monitor: 'val_acc', 'val_loss'
    direction='max',            # Optimization direction: 'max', 'min'
    patience=50,                # Early stopping patience (epochs)
    abort_condition=0.01,       # Minimum improvement threshold
    lr_scheduler=lr_scheduler,  # Learning rate scheduler config
)
```

### Early Stopping
- **Patience**: Number of epochs to wait without improvement
- **Abort Condition**: Minimum improvement required to reset patience counter
- **Direction**: Whether to maximize ('max') or minimize ('min') the monitored metric

### Model Checkpointing
- Automatically saves best model when validation metric improves
- Removes previous checkpoint to save disk space
- Saves both model state and optimizer state for resuming training

## Usage
Choose or extend a runner to control your experiment workflow. Runners are selected in the config files and handle the complete training pipeline from data loading to result reporting.

### Output Structure
```
work_dirs/
├── <config_name>/
│   └── run_<timestamp>/
│       ├── history_fold_*.csv    # Training history per fold
│       ├── fold_*_<metric>_*.pth # Best model checkpoints
│       └── config.yaml           # Saved configuration
```

---
