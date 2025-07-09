# GraphFW.modules

This module contains model definitions for graph neural networks (GNNs) and related architectures.

## Contents
- `base_models/`: Directory for base GNN model implementations (e.g., GCN, GIN).
- `__init__.py`: Module initialization.

## Usage
Register your custom models here to make them available for training via the config system.

## Module Registration
GraphFW uses a registry system to make models, datasets, runners, and other components easily configurable and extensible. Registration allows you to add new modules (e.g., models) and reference them by name in your config files, without changing the core codebase.

To register a model, use the `@MODULES.register_module(type='YourModelName')` decorator above your class definition:

```python
from GraphFW.build import MODULES

@MODULES.register_module(type='MyCustomModel')
class MyCustomModel(nn.Module):
    ...
```

You can then use `'type': 'MyCustomModel'` in your config to select this model.

## Available Registries
- `MODULES`: For GNN model architectures (in `modules/`).
- `DATASETS`: For dataset loaders (in `datasets/`).
- `RUNNERS`: For training/validation/test workflow classes (in `runners/`).
- `OPTIMIZERS`: For optimizer classes.
- `EVALUATORS`: For evaluation logic.

This registry system makes the framework modular and easy to extend for new research or experiments.

---
