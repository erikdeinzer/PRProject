import torch
import torch_geometric as pyg
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from typing import Dict, Any
import os
from registry import TRANSFORMS, build_module, build_modules_list
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



class DatasetLoader:
    def __init__(self, name: str, root: str = "./data", transforms=None, pre_transforms=None) -> None:
        self.name = name
        self.root = root
        self.transforms = build_modules_list(transforms, TRANSFORMS, compose_fn=T.Compose) if transforms else None
        self.pre_transforms = build_modules_list(pre_transforms, TRANSFORMS, compose_fn=T.Compose) if pre_transforms else None
        self.dataset = TUDataset(root=self.root, name=self.name, transform=self.transforms, pre_transform=self.pre_transforms)
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> Dict[str, Any]:
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
        table = soup.find("table")
        if not table:
            raise RuntimeError("Could not find datasets table on the page.")


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
        return self.metadata

    def describe(self) -> None:
        print(f"Dataset: {self.name}")
        for k, v in self.metadata.items():
            print(f"  {k}: {v}")
            
    def get_dataset(self) -> TUDataset:
        return self.dataset

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Data:
        return self.dataset[idx]
    
    def __repr__(self) -> str:
        return f"DatasetModule(name={self.name}, num_graphs={len(self.dataset)})"