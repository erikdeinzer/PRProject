from registry import RUNNERS, MODELS, OPTIMIZERS, build_module, build_modules_list
from .base_runner import BaseRunner
from sklearn.model_selection import train_test_split
from utils import live_train_bar
from torch_geometric.loader import DataLoader
import torch


@RUNNERS.register(type='SplitRunner')
class SplitRunner(BaseRunner):
    """
    SplitRunner is a runner that trains a model on a fixed train/test split.
    """

    def __init__(
        self,
        train_ratio=0.8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.train_ratio = train_ratio
        
        self.model = build_module(self.model_config, MODELS)
        self.optimizer = build_module(self.train_config['optimizer'], OPTIMIZERS, params=self.model.parameters())

        # Create split
        y = [data.y.item() for data in self.dataset]
        self.train_idx, self.test_idx = train_test_split(
            range(len(self.dataset)),
            train_size=self.train_ratio,
            stratify=y,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        if len(self.dataset) < 500:
            print(f"Warning: Dataset is small ({len(self.dataset)} samples). Consider using K-Fold Cross-Validation(KFoldRunner) for better evaluation.")

    def run(self):
        """
        Train and evaluate the model on the train/test split.
        """
        print(f"\nTraining model on {len(self.train_idx)} samples, testing on {len(self.test_idx)} samples.\n")

        train_loader = DataLoader(
            [self.dataset[i] for i in self.train_idx],
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            [self.dataset[i] for i in self.test_idx],
            batch_size=self.batch_size
        )
        acc = 0
        for epoch in range(1, self.train_epochs + 1):
            loss = self.train(self.model, train_loader, self.optimizer, self.device)
            if epoch % self.val_interval == 0:
                acc, val_loss = self.test(self.model, test_loader, self.device)
            if self.log_interval and epoch % self.log_interval == 0:
                live_train_bar(epoch=epoch, total_epochs=self.train_epochs, acc=acc, loss=loss, val_loss=val_loss)

        print("\nFinal evaluation:")
        final_acc, _ = self.test(self.model, test_loader, self.device)
        print(f"Test accuracy: {final_acc:.4f}")