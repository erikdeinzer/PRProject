from typing import Callable, List, Union
import torch_geometric.transforms as T
import inspect
import torch_geometric as pyg
import torch_geometric.nn as nn
import torch


class Registry():
    """
    A simple registry to register and retrieve objects by name.
    This is useful for managing different components and simplifies the process of
    instantiating them based on configuration files.
    The registry can be used to register models, optimizers, and other components
    """
    def __init__(self, name='default'):
        self.name = name
        self._registry = {}

    def register(self, obj=None, *, type=None):
        if obj is None:
            return lambda obj: self.register(obj, type=type)
        key = type or obj.__name__
        if key in self._registry:
            raise ValueError('Duplicate registrations with same name not possible')
        self._registry[key]=obj
        return obj
    
    def get(self, type):
        if type not in self._registry:
            raise KeyError(f"'{type}' not found in the '{self.name}' registry.")
        obj = self._registry.get(type)
        return obj
    
    def all(self):
        return self._registry
    
    def __contains__(self, type):
        return type in self._registry

    def __len__(self):
        return len(self._registry)
    
def build_module(cfg: dict, registry, **kwargs) -> object:
    """
    Build a module from a configuration dictionary.
    Args:
        cfg: Dictionary containing the configuration for the module.
        registry: The registry to use for building the module.
        **kwargs: Additional keyword arguments to pass to the module constructor.
    Returns:
        An instance of the module.
    """
    cfg = cfg.copy()
    cfg.update(kwargs)
    cls = registry.get(cfg.pop('type'))
    return cls(**cfg)

def build_modules_list(cfg: List[dict], registry:Registry, compose_fn:Callable | None = None, **kwargs) -> List[object]:
    """
    Build a list of modules from a list of configuration dictionaries.

    Args:
        cfg: List of dictionaries containing the configuration for each module.
        registry: The registry to use for building the modules.
        compose_fn: A function to compose the modules together.
        **kwargs: Additional keyword arguments to pass to the module constructors.

    Returns:
        A tuple of instantiated modules
    """
    modules = []
    for module_cfg in cfg:
        module = build_module(module_cfg, registry, **kwargs)
        modules.append(module)
    if compose_fn is None:
        return tuple(modules)
    else:
        compose_fn(modules)
    
MODELS = Registry('MODELS')
EVALUATIONS = Registry('EVALUATIONS')
TRANSFORMS = Registry('TRANSFORMS')
OPTIMIZERS = Registry('OPTIMIZERS')
RUNNERS = Registry('RUNNERS')

def register_from_module(module, registry, base_class=None, verbose=False, filter_fn=None):
    """
    Register classes from a module to a registry.
    Args:
        module: The module to register classes from.
        registry: The registry to register the classes to.
        base_class: The base class that the registered classes should inherit from.
        verbose: Whether to print the registration process.
        filter_fn: A function to filter the classes to be registered.
    """

    for name in dir(module):  # Use dir to get full visible symbol list
        try:
            obj = getattr(module, name)  # This triggers lazy imports
        except Exception:
            continue
        if not inspect.isclass(obj):
            continue
        try:
            if base_class is not None and not issubclass(obj, base_class):
                continue
        except TypeError:
            continue
        if filter_fn and not filter_fn(obj):
            continue
        try:
            inspect.signature(obj)  # Just test it's introspectable
            registry.register(obj)
            if verbose:
                print(f"[{registry.name}] Registered: {name}")
        except Exception:
            continue
# Register PyTorch Geometric models
register_from_module(pyg.nn, MODELS, base_class=torch.nn.Module, verbose=False)
# Register PyTorch Geometric transforms
register_from_module(T, TRANSFORMS, base_class=T.BaseTransform, verbose=False)
# Register PyTorch optimizers
register_from_module(torch.optim, OPTIMIZERS, base_class=torch.optim.Optimizer, verbose=False)





