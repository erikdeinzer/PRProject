import argparse
import importlib.util
import sys
import os
import torch
from GraphFW.build import build_module, RUNNERS



def load_config_from_pyfile(pyfile) -> dict:
    pyfile = os.path.abspath(pyfile)
    modulename = os.path.splitext(os.path.basename(pyfile))[0]
    spec = importlib.util.spec_from_file_location(modulename, pyfile)
    if spec is None:
        raise ImportError(f"Could not load spec for {pyfile}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Convert module contents to dictionary, excluding built-ins
    config_dict = {
        k: v for k, v in vars(module).items()
        if not k.startswith("__")
    }
    return config_dict


def main():
    parser = argparse.ArgumentParser(description="Train a GNN model using a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to config .py file')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g. cuda, cpu)')
    args = parser.parse_args()

    try:
        config = load_config_from_pyfile(args.config)
    except Exception as e:
        print(f"Error importing config file: {e}")
        sys.exit(1)

    # Standard Folder
    if 'work_dir' not in config:
        config['work_dir'] = 'work_dirs/' + os.path.splitext(os.path.basename(args.config))[0]
    
    runner_cfg = config.pop('runner', None)

    runner = build_module(
        module = runner_cfg,
        **config,
        registry = RUNNERS,
        device=args.device if args.device else 'cpu',
    )

    runner.run()

if __name__ == "__main__":
    main()
