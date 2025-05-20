import torch
import torch.nn.functional as F
class BaseRunner():
    """
    Base class for all runners.
    """

    def __init__(self, 
                 dataset, 
                 model_config, 
                 train_config, 
                 train_epochs=100,
                 val_interval=1, 
                 logging_config=None,
                 random_state=None,
                 shuffle=True,
                 **kwargs):
        
        """
        Initialize the runner with a model, data, and configuration.

        Args:
            model_config: The model config to be used.
            dataset: The data to be used.
            config: The configuration for the runner.
        """
        self.dataset = dataset

        if isinstance(model_config, dict):
            if model_config['in_channels'] == 'auto':
                # Automatically set the in_channels to the number of features in the dataset
                model_config['in_channels'] = dataset.num_features
        else:
            raise ValueError("model_config should be a dictionary with at least 'type' key.")

        self.model_config = model_config

        self.train_config = train_config
        self.logging_config = logging_config
        self.train_epochs = train_epochs
        self.log_interval = logging_config.get('log_interval', 10) if logging_config else 10
        self.val_interval = val_interval

        self.random_state = random_state
        self.shuffle = shuffle


        self.batch_size = train_config.get('batch_size', 1)
        self.train_epochs = train_config.get('train_epochs', 100)

        self.describe()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


    def run(self):
        """
        Run the model on the data.

        Returns:
            The result of the run.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def train(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test(self, model, loader, device):
        """
        Print out the accuracy of the model on the test set as well as the loss.
        """
        model.eval()
        correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            total_loss += F.cross_entropy(out, data.y)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())

        acc = correct / len(loader.dataset)
        val_loss = total_loss / len(loader)
        return acc, val_loss
    
    def describe(self):
        """
        Describe the runner.
        """
        print(f"Runner: {self.__class__.__name__}")
        print(f"Model: {self.model_config['type']}")
        print(f"Dataset: {self.dataset.__class__.__name__}")
        print(f"Train epochs: {self.train_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Log interval: {self.log_interval}")
        print(f"Validation interval: {self.val_interval}")

        print("-" * 50)
        print('Dataset Info:')
        print(f"Number of graphs: {len(self.dataset)}")
        print(f"Number of features: {self.dataset.num_features}")
        print(f"Number of classes: {self.dataset.num_classes}")

        print("-" * 50)
        print('Model Info:')
        print(f"Model config: {self.model_config}")