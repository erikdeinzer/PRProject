import torch
import torch.nn.functional as F
class BaseRunner():
    """
    Base class for all runners.

    The runners are responsible for training and evaluating the models.
    It is designed for Graph Classification tasks.
    """

    def __init__(self, 
                 dataset_loader, 
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
            dataset: The dataset to be used.
            model_config: The configuration for the model.
            train_config: The configuration for the training.
            train_epochs: Number of epochs to train the model.
            val_interval: Interval for validation.
            logging_config: Configuration for logging.
            random_state: Random state for reproducibility.
            shuffle: Whether to shuffle the data.
        """
        self.dataset_loader = dataset_loader
        self.dataset = self.dataset_loader.get_dataset()

        if isinstance(model_config, dict):
            if model_config['in_channels'] == 'auto':
                # Automatically set the in_channels to the number of features in the dataset
                model_config['in_channels'] = self.dataset.num_features
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
        """
        Train the model on the data.
        Args:
            model: The model to be trained.
            loader: The data loader for the training data.
            optimizer: The optimizer for the model.
            device: The device to run the model on.
        """
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
        Test the model on the data.
        Args:
            model: The model to be tested.
            loader: The data loader for the test data.
            device: The device to run the model on.
        """
        model.eval()
        correct = 0
        total_loss = 0
        total_samples = 0

        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total_samples += data.y.size(0)

        acc = correct / total_samples
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
        self.dataset_loader.describe()

        print("-" * 50)
        print('Model Info:')
        print(f"Model config: {self.model_config}")