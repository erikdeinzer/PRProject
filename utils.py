import sys
def live_train_bar(epoch, total_epochs, loss=None, val_loss=None, acc=None, fold=None, bar_width=10):
        progress = epoch / total_epochs
        filled_len = int(progress * bar_width)
        bar = "█" * filled_len + '-' * (bar_width - filled_len)
        s = ''
        if fold is not None: s += f"Fold {fold} | "
        s += f"Epoch {epoch:03d}/{total_epochs} | [{bar}] | "
        if loss is not None: s += f"Train Loss: {loss:.4f} | "
        if val_loss is not None: s += f"Val Loss: {val_loss:.4f} | "
        if acc is not None: s += f"Acc: {acc:.4f} | "
        sys.stdout.write('\r' + s)
        sys.stdout.flush()