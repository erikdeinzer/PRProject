from registry import RUNNERS, MODELS, OPTIMIZERS, build_module, build_modules_list
from .base_runner import BaseRunner
from sklearn.model_selection import StratifiedKFold
from utils import live_train_bar
from torch_geometric.loader import DataLoader
import torch
import torch_geometric.nn as nn

@RUNNERS.register(type='KFoldRunner')
class KFoldRunner(BaseRunner):
    """
    CVFoldRunner is a specialized runner for K-Fold Cross-Validation.
    It inherits from the BaseRunner class.
    """

    def __init__(self, n_splits=10, fold_epochs=50, **kwargs):
        """
        Initialize the CVFoldRunner with a model, data, and configuration.

        Args:
            model: The model to be used.
            data: The data to be used.
            config: The configuration for the runner.
        """
        super().__init__(**kwargs)

        self.n_splits = n_splits
        self.cv_models = [build_module(self.model_config, MODELS) for _ in range(n_splits)]
        self.cv_optimizers = [build_module(self.train_config['optimizer'], OPTIMIZERS, params=model.parameters()) for model in self.cv_models]

        self.model = build_modules_list(self.model_config, MODELS)
        self.optimizer = build_module(self.train_config['optimizer'], OPTIMIZERS, params=self.model.parameters())

        
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        self.fold_epochs = fold_epochs

        if len(self.dataset) > 1000:
            print(f"Warning: Dataset is large ({len(self.dataset)} samples). Consider using a smaller number of folds for faster training or use Train Test Split (SplitRunner).")

    
    def run(self):
        """
        Run the model on the data using K-Fold Cross-Validation.

        Returns:
            The result of the run.
        """
        accuracies = []
        y = [data.y.item() for data in self.dataset]

        for fold, (_model, _optimizer, (train_idx, test_idx)) in enumerate(zip(self.cv_models, self.cv_optimizers, self.skf.split(self.dataset, y))):
            train_loader = DataLoader([self.dataset[i] for i in train_idx], batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader([self.dataset[i] for i in test_idx], batch_size=self.batch_size)

            
            for epoch in range(1, self.fold_epochs + 1):
                loss = self.train(_model, train_loader, _optimizer, self.device)
                acc, val_loss = self.test(_model, test_loader, self.device)
                if self.log_interval and epoch % self.log_interval == 0:
                    live_train_bar(fold=fold, epoch=epoch, total_epochs=self.fold_epochs, acc=acc, loss=loss, val_loss=val_loss)
                    
            print()
            accuracies.append(acc)

        # Summary
        avg_acc = sum(accuracies) / len(accuracies)
        std_acc = (sum((x - avg_acc) ** 2 for x in accuracies) / len(accuracies)) ** 0.5

        print(f"\nCross-validation accuracy over 10 folds: {avg_acc:.4f} ± {std_acc:.4f}")
        # Final model on full dataset
    
    def train_full(self):
        """
        Train the final model on the full dataset.
        """
        print(f"\nTraining final model on full dataset...\n")
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


        for epoch in range(1, self.train_epochs + 1):
            final_loss = self.train(self.model, loader, self.optimizer, self.device)
            if self.log_interval and epoch % self.log_interval == 0:
                live_train_bar(epoch=epoch, total_epochs=self.train_epochs, loss=final_loss)

        print("\nFinal training complete.")

