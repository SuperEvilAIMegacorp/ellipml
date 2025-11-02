"""
Data loading and preprocessing utilities for Elliptic datasets
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


class EllipticDataLoader:
    """Load and preprocess Elliptic dataset (transactions only)"""

    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files"""
        print("Loading Elliptic transaction data...")

        tx_features = pd.read_csv(f'{self.data_dir}/ellipticpp/txs_features.csv')
        tx_classes = pd.read_csv(f'{self.data_dir}/ellipticpp/txs_classes.csv')
        tx_edges = pd.read_csv(f'{self.data_dir}/ellipticpp/txs_edgelist.csv')

        print(f"  Transactions: {len(tx_features):,}")
        print(f"  Edges: {len(tx_edges):,}")
        print(f"  Features per tx: {tx_features.shape[1]-1}")

        return tx_features, tx_classes, tx_edges

    def prepare_graph_data(self,
                          use_only_labeled: bool = False,
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          use_fraud_patterns: bool = False) -> Data:
        """
        Prepare PyTorch Geometric Data object

        Args:
            use_only_labeled: If True, only include labeled nodes in graph
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (test = 1 - train - val)
            use_fraud_patterns: If True, augment features with fraud pattern scores

        Returns:
            PyTorch Geometric Data object
        """
        tx_features, tx_classes, tx_edges = self.load_raw_data()

        tx_data = tx_features.merge(tx_classes, on='txId', how='left')

        feature_cols = [c for c in tx_data.columns
                       if c not in ['txId', 'class']]
        X = tx_data[feature_cols].values.astype(np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(f"\nFeature matrix shape: {X.shape}")

        node_ids = tx_data['txId'].values
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        y = tx_data['class'].fillna(3).astype(int).values
        y = y - 1

        # Create masks for labeled vs unknown
        labeled_mask = (y != 2)

        print(f"\nLabel distribution:")
        print(f"  Illicit (0): {(y == 0).sum():,}")
        print(f"  Licit (1): {(y == 1).sum():,}")
        print(f"  Unknown (2): {(y == 2).sum():,}")

        # Build edge index
        edge_list = []
        for _, row in tx_edges.iterrows():
            src_id, dst_id = row['txId1'], row['txId2']
            if src_id in node_to_idx and dst_id in node_to_idx:
                src_idx = node_to_idx[src_id]
                dst_idx = node_to_idx[dst_id]
                edge_list.append([src_idx, dst_idx])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        print(f"\nEdge index shape: {edge_index.shape}")

        # Convert to PyTorch tensors
        x = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Augment with fraud pattern features if requested
        if use_fraud_patterns:
            print("\nComputing fraud pattern features...")
            from fraud_patterns import augment_features_with_patterns

            timestamps = tx_data['Time step'].values
            x = augment_features_with_patterns(x, edge_index, timestamps)
            print(f"Augmented feature shape: {x.shape}")

        # Create train/val/test splits on LABELED data only
        labeled_indices = np.where(labeled_mask)[0]

        train_idx, temp_idx = train_test_split(
            labeled_indices,
            train_size=train_ratio,
            stratify=y[labeled_mask],
            random_state=42
        )

        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            stratify=y[temp_idx],
            random_state=42
        )

        # Create masks
        train_mask = torch.zeros(len(y), dtype=torch.bool)
        val_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        print(f"\nData splits:")
        print(f"  Train: {train_mask.sum():,} ({train_mask.sum()/labeled_mask.sum()*100:.1f}%)")
        print(f"  Val: {val_mask.sum():,} ({val_mask.sum()/labeled_mask.sum()*100:.1f}%)")
        print(f"  Test: {test_mask.sum():,} ({test_mask.sum()/labeled_mask.sum()*100:.1f}%)")
        print(f"  Unlabeled: {(~labeled_mask).sum():,}")

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y_tensor,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        return data


class EllipticPlusPlusDataLoader:
    """Load and preprocess Elliptic++ dataset (wallets + transactions)"""

    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir

    def load_heterogeneous_graph(self,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15) -> HeteroData:
        """
        Load complete heterogeneous graph with transactions and wallets

        Returns:
            PyTorch Geometric HeteroData object
        """
        print("="*80)
        print("LOADING HETEROGENEOUS GRAPH (Transactions + Wallets)")
        print("="*80)

        # Load transaction data
        print("\n[1/5] Loading transaction data...")
        tx_features = pd.read_csv(f'{self.data_dir}/ellipticpp/txs_features.csv')
        tx_classes = pd.read_csv(f'{self.data_dir}/ellipticpp/txs_classes.csv')
        tx_edges = pd.read_csv(f'{self.data_dir}/ellipticpp/txs_edgelist.csv')

        # Load wallet data
        print("[2/5] Loading wallet data...")
        wallet_features = pd.read_csv(f'{self.data_dir}/ellipticpp/wallets_features_classes_combined.csv')
        wallet_classes = pd.read_csv(f'{self.data_dir}/ellipticpp/wallets_classes.csv')
        wallet_edges = pd.read_csv(f'{self.data_dir}/ellipticpp/AddrAddr_edgelist.csv')

        # Load bipartite edges
        print("[3/5] Loading bipartite edges...")
        addr_to_tx = pd.read_csv(f'{self.data_dir}/ellipticpp/AddrTx_edgelist.csv')
        tx_to_addr = pd.read_csv(f'{self.data_dir}/ellipticpp/TxAddr_edgelist.csv')

        print(f"\nDataset sizes:")
        print(f"  Transactions: {len(tx_classes):,}")
        print(f"  Wallets: {len(wallet_classes):,}")
        print(f"  Tx→Tx edges: {len(tx_edges):,}")
        print(f"  Wallet→Wallet edges: {len(wallet_edges):,}")
        print(f"  Wallet→Tx edges: {len(addr_to_tx):,}")
        print(f"  Tx→Wallet edges: {len(tx_to_addr):,}")

        # Process transaction features
        print("\n[4/5] Processing node features...")
        tx_data = tx_features.merge(tx_classes, on='txId', how='left')
        tx_feat_cols = [c for c in tx_data.columns
                       if c not in ['txId', 'class']]
        tx_X = tx_data[tx_feat_cols].values.astype(np.float32)

        tx_X = np.nan_to_num(tx_X, nan=0.0, posinf=0.0, neginf=0.0)

        from sklearn.preprocessing import StandardScaler
        tx_scaler = StandardScaler()
        tx_X = tx_scaler.fit_transform(tx_X)

        tx_node_ids = tx_data['txId'].values
        tx_node_to_idx = {node_id: idx for idx, node_id in enumerate(tx_node_ids)}

        tx_y = tx_data['class'].fillna(3).astype(int).values
        tx_y = tx_y - 1

        # Process wallet features (use unique wallets, aggregate temporal instances)
        wallet_unique = wallet_features.groupby('address').agg({
            'num_txs_as_sender': 'sum',
            'num_txs_as receiver': 'sum',
            'total_txs': 'sum',
            'lifetime_in_blocks': 'max',
            'num_timesteps_appeared_in': 'max',
            'btc_transacted_total': 'sum',
            'btc_sent_total': 'sum',
            'btc_received_total': 'sum',
            'class': 'first'  # Class should be same across time steps
        }).reset_index()

        # Select key wallet features
        wallet_feat_cols = [
            'num_txs_as_sender', 'num_txs_as receiver', 'total_txs',
            'lifetime_in_blocks', 'num_timesteps_appeared_in',
            'btc_transacted_total', 'btc_sent_total', 'btc_received_total'
        ]
        wallet_X = wallet_unique[wallet_feat_cols].values.astype(np.float32)

        # Normalize wallet features (they're raw values)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        wallet_X = scaler.fit_transform(wallet_X)

        # Create wallet node mapping
        wallet_addresses = wallet_unique['address'].values
        wallet_addr_to_idx = {addr: idx for idx, addr in enumerate(wallet_addresses)}

        # Wallet labels
        # NOTE: Elliptic++ only has licit (1) and unknown (2/3) wallets, NO illicit labels
        # Map: 1 (licit) → 0, 2/3 (unknown) → 1 for binary classification
        wallet_y = (wallet_unique['class'].values > 1).astype(np.int64)  # 1→0 (licit), 2/3→1 (unknown)

        print(f"\nTransaction features: {tx_X.shape}")
        print(f"Wallet features: {wallet_X.shape}")

        # Build edge indices
        print("\n[5/5] Building edge indices...")

        # Transaction → Transaction
        tx_tx_edges = []
        for _, row in tx_edges.iterrows():
            if row['txId1'] in tx_node_to_idx and row['txId2'] in tx_node_to_idx:
                tx_tx_edges.append([tx_node_to_idx[row['txId1']], tx_node_to_idx[row['txId2']]])
        tx_tx_edge_index = torch.tensor(tx_tx_edges, dtype=torch.long).t().contiguous() if tx_tx_edges else torch.empty((2, 0), dtype=torch.long)

        # Wallet → Wallet (sample to reduce memory)
        print("  Sampling wallet→wallet edges (200k)...")
        wallet_edges_sample = wallet_edges.sample(n=min(200000, len(wallet_edges)), random_state=42)
        wallet_wallet_edges = []
        for _, row in wallet_edges_sample.iterrows():
            if row['input_address'] in wallet_addr_to_idx and row['output_address'] in wallet_addr_to_idx:
                wallet_wallet_edges.append([
                    wallet_addr_to_idx[row['input_address']],
                    wallet_addr_to_idx[row['output_address']]
                ])
        wallet_wallet_edge_index = torch.tensor(wallet_wallet_edges, dtype=torch.long).t().contiguous() if wallet_wallet_edges else torch.empty((2, 0), dtype=torch.long)

        # Wallet → Transaction
        wallet_tx_edges = []
        for _, row in addr_to_tx.iterrows():
            if row['input_address'] in wallet_addr_to_idx and row['txId'] in tx_node_to_idx:
                wallet_tx_edges.append([
                    wallet_addr_to_idx[row['input_address']],
                    tx_node_to_idx[row['txId']]
                ])
        wallet_tx_edge_index = torch.tensor(wallet_tx_edges, dtype=torch.long).t().contiguous() if wallet_tx_edges else torch.empty((2, 0), dtype=torch.long)

        # Transaction → Wallet
        tx_wallet_edges = []
        for _, row in tx_to_addr.iterrows():
            if row['txId'] in tx_node_to_idx and row['output_address'] in wallet_addr_to_idx:
                tx_wallet_edges.append([
                    tx_node_to_idx[row['txId']],
                    wallet_addr_to_idx[row['output_address']]
                ])
        tx_wallet_edge_index = torch.tensor(tx_wallet_edges, dtype=torch.long).t().contiguous() if tx_wallet_edges else torch.empty((2, 0), dtype=torch.long)

        print(f"\nEdge counts:")
        print(f"  Tx→Tx: {tx_tx_edge_index.shape[1]:,}")
        print(f"  Wallet→Wallet: {wallet_wallet_edge_index.shape[1]:,}")
        print(f"  Wallet→Tx: {wallet_tx_edge_index.shape[1]:,}")
        print(f"  Tx→Wallet: {tx_wallet_edge_index.shape[1]:,}")

        # Create HeteroData object
        data = HeteroData()

        # Add transaction nodes
        data['transaction'].x = torch.tensor(tx_X, dtype=torch.float)
        data['transaction'].y = torch.tensor(tx_y, dtype=torch.long)

        # Add wallet nodes
        data['wallet'].x = torch.tensor(wallet_X, dtype=torch.float)
        data['wallet'].y = torch.tensor(wallet_y, dtype=torch.long)

        # Add edges
        data['transaction', 'flows_to', 'transaction'].edge_index = tx_tx_edge_index
        data['wallet', 'sends_to', 'wallet'].edge_index = wallet_wallet_edge_index
        data['wallet', 'inputs_to', 'transaction'].edge_index = wallet_tx_edge_index
        data['transaction', 'outputs_to', 'wallet'].edge_index = tx_wallet_edge_index

        # Create train/val/test masks for both node types
        print("\n[6/6] Creating train/val/test splits...")

        # Transaction splits
        tx_labeled = (tx_y != 2)
        tx_labeled_indices = np.where(tx_labeled)[0]

        tx_train_idx, tx_temp_idx = train_test_split(
            tx_labeled_indices, train_size=train_ratio,
            stratify=tx_y[tx_labeled], random_state=42
        )
        val_ratio_adj = val_ratio / (1 - train_ratio)
        tx_val_idx, tx_test_idx = train_test_split(
            tx_temp_idx, train_size=val_ratio_adj,
            stratify=tx_y[tx_temp_idx], random_state=42
        )

        data['transaction'].train_mask = torch.zeros(len(tx_y), dtype=torch.bool)
        data['transaction'].val_mask = torch.zeros(len(tx_y), dtype=torch.bool)
        data['transaction'].test_mask = torch.zeros(len(tx_y), dtype=torch.bool)
        data['transaction'].train_mask[tx_train_idx] = True
        data['transaction'].val_mask[tx_val_idx] = True
        data['transaction'].test_mask[tx_test_idx] = True

        # Wallet splits
        wallet_labeled = (wallet_y != 2)
        wallet_labeled_indices = np.where(wallet_labeled)[0]

        wallet_train_idx, wallet_temp_idx = train_test_split(
            wallet_labeled_indices, train_size=train_ratio,
            stratify=wallet_y[wallet_labeled], random_state=42
        )
        wallet_val_idx, wallet_test_idx = train_test_split(
            wallet_temp_idx, train_size=val_ratio_adj,
            stratify=wallet_y[wallet_temp_idx], random_state=42
        )

        data['wallet'].train_mask = torch.zeros(len(wallet_y), dtype=torch.bool)
        data['wallet'].val_mask = torch.zeros(len(wallet_y), dtype=torch.bool)
        data['wallet'].test_mask = torch.zeros(len(wallet_y), dtype=torch.bool)
        data['wallet'].train_mask[wallet_train_idx] = True
        data['wallet'].val_mask[wallet_val_idx] = True
        data['wallet'].test_mask[wallet_test_idx] = True

        print(f"\nTransaction splits:")
        print(f"  Train: {data['transaction'].train_mask.sum():,}")
        print(f"  Val: {data['transaction'].val_mask.sum():,}")
        print(f"  Test: {data['transaction'].test_mask.sum():,}")

        print(f"\nWallet splits:")
        print(f"  Train: {data['wallet'].train_mask.sum():,}")
        print(f"  Val: {data['wallet'].val_mask.sum():,}")
        print(f"  Test: {data['wallet'].test_mask.sum():,}")

        print("\n" + "="*80)
        print("Heterogeneous graph loaded successfully!")
        print("="*80)

        return data


if __name__ == "__main__":
    # Test data loading
    print("Testing Elliptic data loader...")
    loader = EllipticDataLoader()
    data = loader.prepare_graph_data()
    print(f"\nPyG Data object: {data}")

    print("\n" + "="*80)
    print("Testing Elliptic++ heterogeneous data loader...")
    hetero_loader = EllipticPlusPlusDataLoader()
    hetero_data = hetero_loader.load_heterogeneous_graph()
    print(f"\nPyG HeteroData object: {hetero_data}")
