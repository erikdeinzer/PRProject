from GraphFW.build import RUNNERS, MODULES, OPTIMIZERS, SCHEDULERS, build_module
from .base_runner import BaseRunner
from .utils.progress_bar import progress_bar
from torch_geometric.loader import DataLoader
import torch
from sklearn.model_selection import train_test_split

import os

@RUNNERS.register_module(type='SplitRunner')
class SplitRunner(BaseRunner):
    """
    SplitRunner for train/test split, using the BaseRunner structure.
    """
    def __init__(self, train_ratio=0.8, **kwargs):
        super().__init__(**kwargs)
        self.train_ratio = train_ratio

        self.model = build_module(self.model_cfg, MODULES)
        self.optimizer = build_module(self.optim_cfg, OPTIMIZERS, params=self.model.parameters())

        if self.scheduler_cfg is not None:
            scheduler_cfg = self.scheduler_cfg.copy()
            self.scheduler = build_module(scheduler_cfg, SCHEDULERS, optimizer=self.optimizer)

        indices = list(range(len(self.dataset)))
        train_idx, test_idx = train_test_split(
            indices,
            train_size=self.train_ratio,
            stratify=[data.y.item() for data in self.dataset],
            shuffle=self.shuffle,
            random_state=kwargs.get('seed', None)
        )

        self.train_set = [self.dataset[i] for i in train_idx]
        self.test_set = [self.dataset[i] for i in test_idx]

        if len(indices) < 500:
            print(f"Warning: Dataset is small ({len(indices)} samples). Consider using K-Fold Cross-Validation (KFoldRunner) for better evaluation.")

    
    
    def train(self, start_epoch=None, epochs=None):
        epochs = epochs or self.train_epochs
        start_epoch = start_epoch or self.start_epoch
        
        train_loader = DataLoader(self.train_set, **self.train_dataloader)

        acc = 0
        
        last_file = None
        for epoch in range(start_epoch, epochs + 1):
            print()
            avg_loss = self._train_epoch(self.model, train_loader, self.optimizer, epoch, total_epochs=epochs)

            self.history['train_loss'].append(avg_loss)
            if epoch % self.val_interval == 0:
                acc, val_loss = self.evaluate(model=self.model, data=self.test_set)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(acc)
            
            if self._check_abort():
                print()
                print("\nEarly stopping triggered.")
                break
            if self._check_saving():
                if last_file:
                    os.remove(last_file)    
                filename = f'best_ckpt_{self.metric}_{self.history[self.metric][-1]:.4f}.pth'
                last_file = self.save_model(filename=filename)
            
            self.write_history_to_csv(self.history, filename=f'history.csv')

            if self.scheduler is not None:
                self.scheduler.step()

        
        print("\nFinal evaluation:")
        final_acc, _ = self.evaluate(self.model, self.test_set)
        print(f"Test accuracy: {final_acc:.4f}")
        return final_acc

    def predict(self, loss=True):
        self.model.eval()
        val_loader = DataLoader(self.test_set, **self.val_dataloader)
        preds = []
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                if loss:
                    losses.append(self.criterion(out, batch.y).item())
        return preds, (sum(losses) / len(losses) if losses else None)