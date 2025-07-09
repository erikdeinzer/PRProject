from GraphFW.build import RUNNERS, MODULES, OPTIMIZERS, build_module
from .base_runner import BaseRunner
from .utils.progress_bar import progress_bar
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import torch

@RUNNERS.register_module(type='KFoldRunner')
class KFoldRunner(BaseRunner):
    """
    KFoldRunner for K-Fold Cross-Validation, using the BaseRunner structure.
    """
    def __init__(self, n_splits=10, **kwargs):
        super().__init__(**kwargs)
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        self.models = torch.nn.ModuleList()
        self.optimizers = []

        for _ in range(self.n_splits):
            model = build_module(self.model_cfg, MODULES)
            self.models.append(model)
            optimizer = build_module(self.optim_cfg, OPTIMIZERS, params=model.parameters())
            self.optimizers.append(optimizer)

        if hasattr(self, 'train_data') and len(self.train_data) > 1000:
            print(f"Warning: Dataset is large ({len(self.train_data)} samples). Consider using a smaller number of folds for faster training or use Train Test Split (SplitRunner).")
        
        fold_hist = {
                'train_loss': [],
                'val_loss': [],
                'val_acc': []
            }
        self.history = [fold_hist for _ in range(self.n_splits)]

    def train(self, start_epoch=None, epochs=None):
        epochs = epochs or self.train_epochs
        start_epoch = start_epoch or self.start_epoch
        
        accuracies = []
        y = [data.y.item() for data in self.dataset]
        for i, (train_idx, test_idx) in enumerate(self.skf.split(self.dataset, y)):
            print(f"\nFold {i + 1}/{self.n_splits}")
            train_set = [self.dataset[i] for i in train_idx]
            test_set = [self.dataset[i] for i in test_idx]

            model = self.models[i].to(self.device)
            optimizer = self.optimizers[i]

            train_loader = DataLoader(train_set, **self.train_dataloader)
            acc = 0
            for epoch in range(self.start_epoch, epochs + 1):
                print()
                avg_loss = self._train_epoch(model, train_loader, optimizer, epoch, total_epochs=epochs)

                self.history[i]['train_loss'].append(avg_loss)
                if epoch % self.val_interval == 0:
                    acc, val_loss = self.evaluate(model=model, data=test_set)
                    self.history[i]['val_loss'].append(val_loss)
                    self.history[i]['val_acc'].append(acc)
                
                if self._check_abort(self.history[i]):
                    print("\nEarly stopping triggered.")
                    break
                if self._check_saving(self.history[i]):
                    self.save_model(filename=f'fold_{i + 1}_ckpt_{self.metric}_{self.history[i][self.metric][-1]:.4f}.pth', history=self.history[i], optimizer=optimizer, model=model)
            print()
            accuracies.append(acc)
        avg_acc = sum(accuracies) / len(accuracies)
        std_acc = (sum((x - avg_acc) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
        print(f"\nCross-validation accuracy over {self.n_splits} folds: {avg_acc:.4f} ± {std_acc:.4f}")
        return avg_acc

    def predict(self, data, batch_size=1, loss=True):
        # For KFoldRunner, prediction is not typically used, but can be implemented if needed
        self.model.eval()
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        preds = []
        losses = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                if loss:
                    losses.append(self.criterion(out, batch.y).item())
        return preds, (sum(losses) / len(losses) if losses else None)

