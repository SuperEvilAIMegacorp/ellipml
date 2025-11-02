"""
Graph Neural Network models for Bitcoin fraud detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import HeteroConv, Linear


class GCN(nn.Module):
    """
    Baseline Graph Convolutional Network

    Simple GCN for transaction-only or wallet-only classification
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network

    Uses attention mechanism to weight neighbor importance
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer with multi-head attention
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                     heads=heads, dropout=dropout))

        # Output layer (single head)
        self.convs.append(GATConv(hidden_channels * heads, num_classes,
                                 heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    """
    GraphSAGE model

    Efficient for large graphs, samples neighborhoods
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network

    Handles multiple node types (transactions, wallets) and edge types
    Uses HeteroConv for message passing across different relation types
    """

    def __init__(self,
                 tx_in_channels: int,
                 wallet_in_channels: int,
                 hidden_channels: int = 128,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection layers (make dimensions compatible)
        self.tx_lin = Linear(tx_in_channels, hidden_channels)
        self.wallet_lin = Linear(wallet_in_channels, hidden_channels)

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                # Transaction → Transaction
                ('transaction', 'flows_to', 'transaction'): SAGEConv(hidden_channels, hidden_channels),

                # Wallet → Wallet
                ('wallet', 'sends_to', 'wallet'): SAGEConv(hidden_channels, hidden_channels),

                # Wallet → Transaction (wallet as input)
                ('wallet', 'inputs_to', 'transaction'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),

                # Transaction → Wallet (wallet as output)
                ('transaction', 'outputs_to', 'wallet'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            }, aggr='mean')

            self.convs.append(conv)

        # Output classifiers (separate for each node type)
        self.tx_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

        self.wallet_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x_dict, edge_index_dict):
        # Project input features to same hidden dimension
        x_dict = {
            'transaction': self.tx_lin(x_dict['transaction']),
            'wallet': self.wallet_lin(x_dict['wallet'])
        }

        # Apply heterogeneous convolutions
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                         for key, x in x_dict.items()}

        # Apply separate classifiers
        out_dict = {
            'transaction': self.tx_classifier(x_dict['transaction']),
            'wallet': self.wallet_classifier(x_dict['wallet'])
        }

        return out_dict


class HeteroGNNWithAttention(nn.Module):
    """
    Heterogeneous GNN with attention mechanism

    Uses attention to weight importance of different edge types
    """

    def __init__(self,
                 tx_in_channels: int,
                 wallet_in_channels: int,
                 hidden_channels: int = 128,
                 num_classes: int = 2,
                 num_layers: int = 2,
                 heads: int = 4,
                 dropout: float = 0.5):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        # Input projection
        self.tx_lin = Linear(tx_in_channels, hidden_channels)
        self.wallet_lin = Linear(wallet_in_channels, hidden_channels)

        # Heterogeneous GAT layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_channels if i == 0 else hidden_channels * heads

            conv = HeteroConv({
                ('transaction', 'flows_to', 'transaction'):
                    GATConv(in_dim, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),

                ('wallet', 'sends_to', 'wallet'):
                    GATConv(in_dim, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),

                ('wallet', 'inputs_to', 'transaction'):
                    GATConv((in_dim, in_dim), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),

                ('transaction', 'outputs_to', 'wallet'):
                    GATConv((in_dim, in_dim), hidden_channels, heads=heads, dropout=dropout, add_self_loops=False),
            }, aggr='mean')

            self.convs.append(conv)

        # Output classifiers
        final_dim = hidden_channels * heads

        self.tx_classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )

        self.wallet_classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x_dict, edge_index_dict):
        # Project inputs
        x_dict = {
            'transaction': self.tx_lin(x_dict['transaction']),
            'wallet': self.wallet_lin(x_dict['wallet'])
        }

        # Apply hetero convolutions
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)

            if i < len(self.convs) - 1:
                x_dict = {key: F.elu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                         for key, x in x_dict.items()}

        # Classify
        out_dict = {
            'transaction': self.tx_classifier(x_dict['transaction']),
            'wallet': self.wallet_classifier(x_dict['wallet'])
        }

        return out_dict


def get_model(model_type: str, **kwargs):
    """
    Factory function to get model by name

    Args:
        model_type: One of ['gcn', 'gat', 'sage', 'hetero', 'hetero_attn']
        **kwargs: Model-specific parameters

    Returns:
        Model instance
    """
    model_type = model_type.lower()

    if model_type == 'gcn':
        return GCN(**kwargs)
    elif model_type == 'gat':
        return GAT(**kwargs)
    elif model_type == 'sage' or model_type == 'graphsage':
        return GraphSAGE(**kwargs)
    elif model_type == 'hetero':
        return HeteroGNN(**kwargs)
    elif model_type == 'hetero_attn':
        return HeteroGNNWithAttention(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model instantiation
    print("Testing model creation...")

    print("\n1. GCN")
    model = GCN(in_channels=166, hidden_channels=64, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n2. GAT")
    model = GAT(in_channels=166, hidden_channels=64, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n3. GraphSAGE")
    model = GraphSAGE(in_channels=166, hidden_channels=64, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n4. HeteroGNN")
    model = HeteroGNN(tx_in_channels=166, wallet_in_channels=8, hidden_channels=64, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n5. HeteroGNN with Attention")
    model = HeteroGNNWithAttention(tx_in_channels=166, wallet_in_channels=8, hidden_channels=64, num_classes=2)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nAll models created successfully!")
