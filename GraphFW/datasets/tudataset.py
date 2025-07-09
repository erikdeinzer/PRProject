import torch
import torch_geometric as pyg
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from typing import Dict, Any
import os
from GraphFW.build import TRANSFORMS, DATASETS, build_module
from torch import nn

from typing import Union, List, Callable
import requests
def check_internet_connection(url, timeout=5):
            try:
                response = requests.get(url, timeout=timeout)
                return response.status_code == 200
            except requests.RequestException:
                return False

ONLINE = check_internet_connection('https://chrsmrrs.github.io/datasets/docs/datasets/')
if ONLINE:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        ONLINE = False
        print("No online stats available. Please install requests and BeautifulSoup4 to fetch online dataset statistics.")


@DATASETS.register_module(type='TUDatasetLoader')
class TUDatasetLoader:
    def __init__(self, name: str, root: str = "./data", transforms=None, pre_transforms=None) -> None:
        """
        Initialize the DatasetLoader with a dataset name, root directory, and optional transformations.
        This class loads the dataset using PyTorch Geometric's TUDataset and applies the specified transformations.
        It also extracts metadata about the dataset, including the number of graphs, classes, and node features.


        Args:
            name (str): Name of the dataset to load.
            root (str): Directory to store the dataset.
            transforms (list): List of transformations to apply to the dataset
            pre_transforms (list): List of transformations to apply before loading the dataset.
        """
        self.name = name
        self.root = root
        self.transforms = T.Compose([build_module(t, TRANSFORMS) for t in transforms]) if transforms else None
        self.pre_transforms = T.Compose([build_module(t, TRANSFORMS) for t in pre_transforms]) if pre_transforms else None
        self.dataset = TUDataset(root=self.root, name=self.name, transform=self.transforms, pre_transform=self.pre_transforms)
        self.metadata = self._extract_metadata()
        self.num_features = self.dataset.num_features if hasattr(self.dataset, 'num_features') else None

    def _extract_metadata(self) -> Dict[str, Any]:
        """
        Extracts metadata from the dataset, including the number of graphs, classes, and node features.
        It also checks the average, maximum, and minimum number of nodes in the graphs, and whether the dataset is homogeneous (all graphs have the same number of node features).
        """
        sample = self.dataset[0]
        num_graphs = len(self.dataset)
        num_classes = self.dataset.num_classes
        num_node_features = self.dataset.num_node_features

        graph_sizes = [data.num_nodes for data in self.dataset]
        avg_nodes = sum(graph_sizes) / len(graph_sizes)
        max_nodes = max(graph_sizes)
        min_nodes = min(graph_sizes)

        # Check homogeneity (all graphs same number of node features)
        homogeneous = all(data.x.size(1) == num_node_features for data in self.dataset if data.x is not None)

        stats =  {
            "dataset": self.name,
            "num_graphs": num_graphs,
            "num_classes": num_classes,
            "num_node_features": num_node_features,
            "avg_nodes_per_graph": avg_nodes,
            "max_nodes": max_nodes,
            "min_nodes": min_nodes,
            "is_homogeneous": homogeneous,
        }

        if ONLINE:
            # Fetch online stats if available
            try:
                online_stats = self._get_online_stats(self.name)
                stats.update(online_stats)
            except Exception as e:
                print(f"Error fetching online stats: {e}")
        
        return stats
    def _get_online_stats(self, dataset_name: str) -> dict:
        """
        Fetches the dataset statistics row from the TUDataset docs page and returns a dict of properties.

        Args:
            dataset_name: Name of the dataset as listed on the page (e.g., "AIDS", "alchemy_full").

        Returns:
            A dictionary containing keys:
            - avg_edges (float)
            - node_labels (bool)
            - edge_labels (bool)
            - node_attr (int or None)
            - geometry (str or None)
            - edge_attr (bool)
        """
        
        response = requests.get('https://chrsmrrs.github.io/datasets/docs/datasets/')
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the table row matching the dataset name
        tables = soup.find_all("table")
        if not tables:
            raise ValueError("No tables found on the TUDataset description page.")
        for table in tables:
            
            for row in table.find_all("tr")[2:]:
                cols = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
                if cols and cols[0].lower() == dataset_name.lower():
                    # Remove any brackets or references from number fields
                    return {
                        "avg_edges": float(cols[5]),
                        "node_labels": cols[6] == '+',
                        "edge_labels": cols[7] == '+',
                        "node_attr": cols[8][3:-1] if cols[8].startswith('+') else None,
                        "geometry": cols[9] if cols[9] != '–' else None,
                        "edge_attr": cols[10] == '+'
                    }

        raise ValueError(f"Dataset '{dataset_name}' not found on the TUDataset description https://chrsmrrs.github.io/datasets/docs/datasets/.")
        
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns the metadata of the dataset, including number of graphs, classes, and node features.
        """
        return self.metadata

    def describe(self) -> None:
        """
        Prints a description of the dataset, including the number of graphs, classes, and node features.
        """
        for k, v in self.metadata.items():
            print(f"  {k}: {v}")
            
    def get_dataset(self) -> TUDataset:
        """
        Returns the loaded dataset.
        """
        return self.dataset

    def __len__(self) -> int:
        """
        Returns the number of graphs in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Returns a single graph from the dataset at the specified index.
        Args:
            idx (int): Index of the graph to retrieve.
        Returns:
            Data: The graph data object at the specified index.
        """
        return self.dataset[idx]
    
    def __repr__(self) -> str:
        return f"DatasetModule(name={self.name}, num_graphs={len(self.dataset)})"