# GraphFW.runners

This module contains runner classes that handle the training, validation, and testing loops for graph classification experiments.

## Contents
- `base_runner.py`: Base class for all runners.
- `kfold_runner.py`: K-Fold cross-validation runner.
- `split_runner.py`: Train/validation/test split runner.
- `utils/`: Progress bar and other runner utilities.
- `__init__.py`: Module initialization.

## Usage
Choose or extend a runner to control your experiment workflow. Runners are selected in the config files.

---
