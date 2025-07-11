#!/usr/bin/env python3
"""
Create synthetic graph data for testing model training without internet access.
This creates a simple binary classification dataset with two classes.
"""

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import os

class SyntheticDataset(InMemoryDataset):
    def __init__(self, root, name='SYNTHETIC', num_graphs=1000, num_node_features=4, num_classes=2, transform=None, pre_transform=None):
        self.name = name
        self.num_graphs = num_graphs
        self._num_node_features = num_node_features
        self._num_classes = num_classes
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property 
    def num_node_features(self):
        return self._num_node_features
    
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def raw_file_names(self):
        return ['synthetic_data.txt']  # Dummy file

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Create dummy raw file to satisfy the parent class
        os.makedirs(self.raw_dir, exist_ok=True)
        with open(os.path.join(self.raw_dir, 'synthetic_data.txt'), 'w') as f:
            f.write("Synthetic dataset for testing\n")

    def process(self):
        np.random.seed(42)
        torch.manual_seed(42)
        
        data_list = []
        
        for i in range(self.num_graphs):
            # Create graphs with 10-30 nodes
            num_nodes = np.random.randint(10, 31)
            
            # Create features with better class separation
            x = torch.randn(num_nodes, self._num_node_features)
            
            # Create label first to guide feature generation
            y = torch.tensor([i % 2], dtype=torch.long)  # Alternate classes
            
            # Bias features towards class label for better learnability
            class_bias = 1.0 if y.item() == 1 else -1.0
            x[:, 0] += class_bias * 0.5  # Add class-dependent bias to first feature
            
            # Create edges (random graph with reasonable connectivity)
            edge_prob = 0.3  # Probability of edge between any two nodes
            edges = []
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    if np.random.random() < edge_prob:
                        edges.append([u, v])
                        edges.append([v, u])  # Make undirected
            
            if len(edges) == 0:  # Ensure at least one edge
                edges = [[0, 1], [1, 0]]
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Label is already created above based on alternating pattern
            # This creates a more predictable and learnable dataset
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    # Create synthetic dataset
    dataset = SyntheticDataset('./data/SYNTHETIC')
    print(f"Created synthetic dataset with {len(dataset)} graphs")
    print(f"Number of features: {dataset.num_node_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Check a sample
    sample = dataset[0]
    print(f"Sample graph: {sample.num_nodes} nodes, {sample.num_edges} edges")
    print(f"Sample label: {sample.y.item()}")