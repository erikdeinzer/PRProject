import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from GraphFW.build import build_module, DATASETS, MODULES, OPTIMIZERS, EVALUATORS
from .utils.progress_bar import progress_bar
from torch_geometric.loader import DataLoader

class BaseRunner:
    """
    Base class for all runners.

    The runners are responsible for training and evaluating the models.
    It is designed for Graph Classification tasks.
    """

    def __init__(self, 
                 model, 
                 dataset, 
                 train_dataloader,
                 val_dataloader,
                 optimizer, 
                 test_dataloader=None,
                 work_dir=None, 
                 device='cpu', 
                 seed=None, 
                 patience=15, 
                 abort_condition=0.05, 
                 direction='min', 
                 metric='val_loss', 
                 save_best=True, 
                 train_epochs=None,
                 val_interval=1, 
                 log_interval=1,
                 shuffle=True,
                 start_epoch=1,
                 **kwargs):
        """
        Initialize the runner with configs for model, data, and training.
        Builds the dataset and model internally.
        """
        self.model_cfg = model
        

        # Dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader or {}

        # Configs
        self.dataset_cfg = dataset
        self.optim_cfg = optimizer


        # Set and Save Seed
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError("Seed must be an integer.")
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"Using seed: {seed}")
        else:
            seed = torch.initial_seed() & 0xFFFFFFFF
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"No seed provided, using initial seed: {seed}")

        self.seed = seed

        # Training setup
        self.start_epoch = start_epoch
        self.train_epochs = train_epochs if train_epochs is not None else 10000 #Maximum epochs
        self.log_interval = log_interval
        self.val_interval = val_interval


        self.history = {'train_loss': [], 'val_loss': [],  'val_acc': []}
        self.shuffle = shuffle

        self.device = torch.device(device)

        self.criterion = getattr(model, 'criterion', torch.nn.CrossEntropyLoss())
        ds_loader = build_module(self.dataset_cfg, DATASETS)
        self.dataset = ds_loader.dataset if hasattr(ds_loader, 'dataset') else ds_loader


        # Early Stopping and Model Saving
        self.work_dir = work_dir
        self.patience = patience
        self.abort_condition = abort_condition
        self.direction = direction
        self.metric = metric
        self.save_best = save_best

        if work_dir:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.save_dir = os.path.join(work_dir, f"run_{timestamp}")
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_cfg()
        else:
            self.save_dir = None
        

    def get_train_loader(self):
        return DataLoader(self.train_data, **self.train_dataloader_cfg)

    def get_val_loader(self):
        return DataLoader(self.val_data, **self.val_dataloader_cfg)

    def get_test_loader(self):
        return DataLoader(self.test_data, **self.test_dataloader_cfg)

    def _train_epoch(self, model, train_loader, optimizer, epoch, total_epochs=None):
        model.train()
        total_loss = 0.
        prior_vars = {
            'total_epochs': total_epochs if total_epochs is not None else 'inf',
            'total_iterations': len(train_loader),
        }
        for i, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if self.log_interval and (i % self.log_interval == 0 or i == len(train_loader)):
                    acc = out.argmax(dim=1).eq(batch.y).sum().item() / batch.num_graphs
                    prior_vars.update({'epoch': epoch, 'iteration': i+1})
                    posterior_vars = {'train_loss': total_loss / (i + 1), 'train_acc': acc}
                    progress_bar(prior_vars=prior_vars, posterior_vars=posterior_vars)
        return total_loss / len(train_loader)
    

    def save_cfg(self, filename='config.yaml'):
        import yaml
        cfg = {
            'model': str(self.model_cfg),
            'train_dataloader': self.train_dataloader,
            'val_dataloader': self.val_dataloader,
            'test_dataloader': self.test_dataloader,
            'dataset': str(self.dataset_cfg),
            'optim': str(self.optim_cfg),
            'work_dir': self.save_dir,
            'device': str(self.device),
            'seed': torch.initial_seed(),
        }
        with open(os.path.join(self.save_dir, filename), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    def save_model(self, filename='ckpt.pth', history = None, optimizer=None, model=None):
        history = history or self.history
        optimizer = optimizer or self.optimizer
        model = model or self.model
        torch.save({'metrics': history}, os.path.join(self.save_dir, 'metrics.pth'))
        torch.save({'optimizer_state': optimizer.state_dict()}, os.path.join(self.save_dir, 'optimizer.pth'))
        if hasattr(model, 'save_checkpoint'):
            model.save_checkpoint(path=os.path.join(self.save_dir, filename))
        else:
            torch.save(model.state_dict(), os.path.join(self.save_dir, filename))
        print(f"Model saved to {os.path.join(self.save_dir, filename)}")
        return os.path.join(self.save_dir, filename)


    def run(self, mode='train', epochs=100, start_epoch=1):
        """
        Run the model on the data.

        Returns:
            The result of the run.
        """
        if mode == 'train':
            return self.train(start_epoch, epochs)
        elif mode == 'validation':
            return self.evaluate(self.val_data, batch_size=1)
        elif mode == 'test':
            return self.predict(self.test_data, batch_size=self.batch_size, loss=False)
        else:
            raise ValueError("Mode must be 'train', 'validation', or 'test'.")

    def _check_abort(self, history=None):
        if history is None:
            history = self.history
        values = history.get(self.metric, [])
        if len(values) < self.patience + 1:
            return False
        window = values[-(self.patience + 1):]
        baseline, *rest = window
        if self.direction == 'min':
            if min(rest) <= baseline - self.abort_condition:
                return False
            print(f"\nEarly stopping: no drop ≥{self.abort_condition} in last {self.patience} epochs of {self.metric}.")
            return True
        elif self.direction == 'max':
            if max(rest) >= baseline + self.abort_condition:
                return False
            print(f"\nEarly stopping: no rise ≥{self.abort_condition} in last {self.patience} epochs of {self.metric}.")
            return True
        else:
            raise ValueError("direction must be 'min' or 'max'")
        
    def _check_saving(self, history=None):
        if history is None:
            history = self.history
        values = history.get(self.metric, [])
        if len(values) < 2:
            return False
        if self.direction == 'min':
            return values[-1] < min(values[:-1])
        elif self.direction == 'max':
            return values[-1] > max(values[:-1])
        else:
            raise ValueError("direction must be 'min' or 'max'")
    
    def train(self, epochs, start_epoch=None):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def predict(self, data, batch_size=1, loss=True):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def evaluate(self, model, data):
        """
        Evaluate the model on the given data.

        Args:
            model: The model to evaluate.
            data: The data to evaluate on.
            batch_size: The batch size for evaluation.

        Returns:
            The evaluation results.
        """
        model.eval()
        loader = DataLoader(data, **self.val_dataloader)
        correct = 0
        total = 0
        total_loss = 0.0
        print()

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                progress_bar(
                    prefix='\t Validation',
                    prior_vars={'iteration': i+1, 'total_iterations': len(loader)},
                    posterior_vars={'val_loss': total_loss / (i + 1), 'val_acc': correct / (total + 1)},
                    style='arrow'
                )

        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        return acc, avg_loss
    
    def _count_params(self, model):
        """
        Count the number of parameters in the model.
        """
        # Non-trainable params
        params_fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        # Trainable params
        params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(
            "-"* 50,
            f"\nModel: {model.__class__.__name__}",
            f"\nTrainable parameters: {params_trainable:,} ({params_trainable / 1e6:.2f}M)",
            f"\nNon-trainable parameters: {params_fixed:,} ({params_fixed / 1e6:.2f}M)",
            f"\nTotal parameters: {params_trainable + params_fixed:,} ({(params_trainable + params_fixed) / 1e6:.2f}M)",
            "\n" + "-" * 50  
        )



