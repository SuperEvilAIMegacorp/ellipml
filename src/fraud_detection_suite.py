import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json

from data_loader import EllipticDataLoader, EllipticPlusPlusDataLoader
from models import GCN, GAT, GraphSAGE, HeteroGNN, HeteroGNNWithAttention
from train_utils import train_epoch, evaluate, get_class_weights, EarlyStopping
from structural_features import StructuralFeatureEngineer


class FraudDetectionTestSuite:
    """
    Comprehensive test suite for graph-based fraud detection models
    """

    def __init__(self,
                 data_dir: str = '.',
                 results_dir: str = './results',
                 device: str = None,
                 use_structural_features: bool = False,
                 use_fraud_patterns: bool = False):

        self.data_dir = data_dir
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.use_structural_features = use_structural_features
        self.use_fraud_patterns = use_fraud_patterns
        self.feature_engineer = StructuralFeatureEngineer() if use_structural_features else None

        self.results = []

    def _load_homogeneous_data(self) -> Tuple:
        """
        Load transaction-only data with optional structural feature augmentation
        """
        loader = EllipticDataLoader(self.data_dir)
        data = loader.prepare_graph_data(use_fraud_patterns=self.use_fraud_patterns)

        if self.use_structural_features:
            data.x = self.feature_engineer.augment_features(
                data.x,
                data.edge_index,
                compute_expensive=False
            )

        return data, data.x.size(1)

    def _load_heterogeneous_data(self) -> Tuple:
        """
        Load heterogeneous graph (transactions + wallets)
        """
        loader = EllipticPlusPlusDataLoader(self.data_dir)
        data = loader.load_heterogeneous_graph()

        tx_feat_dim = data['transaction'].x.size(1)
        wallet_feat_dim = data['wallet'].x.size(1)

        if self.use_structural_features:
            data['transaction'].x = self.feature_engineer.augment_features(
                data['transaction'].x,
                data['transaction', 'flows_to', 'transaction'].edge_index,
                compute_expensive=False
            )
            tx_feat_dim = data['transaction'].x.size(1)

        return data, tx_feat_dim, wallet_feat_dim

    def train_model(self,
                   model: nn.Module,
                   data,
                   is_hetero: bool = False,
                   epochs: int = 100,
                   lr: float = 0.01,
                   weight_decay: float = 5e-4,
                   patience: int = 20,
                   loss_type: str = 'focal',
                   class_weight_alpha: float = 0.5,
                   focal_gamma: float = 2.0) -> Dict:
        """
        Train a model and return performance metrics

        Args:
            model: PyTorch model
            data: PyG Data or HeteroData
            is_hetero: Whether model is heterogeneous
            epochs: Maximum training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            loss_type: 'ce' (weighted CE), 'focal', or 'ce_unweighted'
            class_weight_alpha: Scaling factor for class weights (0-1)
            focal_gamma: Gamma parameter for focal loss

        Returns:
            Dictionary with training results and metrics
        """
        from train_utils import FocalLoss, WeightedCrossEntropyLoss

        model = model.to(self.device)
        data = data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = None
        if loss_type != 'ce_unweighted':
            if is_hetero:
                y = data['transaction'].y
            else:
                y = data.y
            class_weights = get_class_weights(y, num_classes=2, alpha=class_weight_alpha).to(self.device)

            if loss_type == 'focal':
                loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            elif loss_type == 'ce':
                loss_fn = WeightedCrossEntropyLoss(class_weights)

        early_stopping = EarlyStopping(patience=patience, mode='max')

        best_val_f1 = 0
        best_epoch = 0
        training_time = 0

        for epoch in range(epochs):
            epoch_start = time.time()

            loss = train_epoch(model, data, optimizer, self.device,
                             is_hetero=is_hetero, loss_fn=loss_fn)

            epoch_time = time.time() - epoch_start
            training_time += epoch_time

            if epoch % 10 == 0:
                val_metrics = evaluate(model, data, self.device,
                                     is_hetero=is_hetero,
                                     mask_name='val_mask',
                                     node_type='transaction' if is_hetero else None)

                if val_metrics['f1_illicit'] > best_val_f1:
                    best_val_f1 = val_metrics['f1_illicit']
                    best_epoch = epoch

                if early_stopping(val_metrics['f1_illicit']):
                    break

        test_metrics = evaluate(model, data, self.device,
                              is_hetero=is_hetero,
                              mask_name='test_mask',
                              node_type='transaction' if is_hetero else None)

        results = {
            'final_epoch': epoch + 1,
            'best_epoch': best_epoch,
            'training_time': training_time,
            'test_metrics': test_metrics,
            'best_val_f1': best_val_f1
        }

        if is_hetero:
            wallet_test_metrics = evaluate(model, data, self.device,
                                          is_hetero=True,
                                          mask_name='test_mask',
                                          node_type='wallet')
            results['wallet_test_metrics'] = wallet_test_metrics

        return results

    def test_gcn(self, data, in_channels: int, config: Dict = None) -> Dict:
        """Test GCN baseline"""
        if config is None:
            config = {'hidden_channels': 128, 'num_layers': 2, 'dropout': 0.5}

        model = GCN(in_channels=in_channels, **config)

        results = self.train_model(model, data, is_hetero=False)
        results['model_name'] = 'GCN'
        results['config'] = config

        return results

    def test_gat(self, data, in_channels: int, config: Dict = None) -> Dict:
        """Test GAT baseline"""
        if config is None:
            config = {'hidden_channels': 128, 'num_layers': 2,
                     'heads': 4, 'dropout': 0.5}

        model = GAT(in_channels=in_channels, **config)

        results = self.train_model(model, data, is_hetero=False)
        results['model_name'] = 'GAT'
        results['config'] = config

        return results

    def test_graphsage(self, data, in_channels: int, config: Dict = None) -> Dict:
        """Test GraphSAGE baseline"""
        if config is None:
            config = {'hidden_channels': 128, 'num_layers': 2, 'dropout': 0.5}

        model = GraphSAGE(in_channels=in_channels, **config)

        results = self.train_model(model, data, is_hetero=False)
        results['model_name'] = 'GraphSAGE'
        results['config'] = config

        return results

    def test_hetero_gnn(self, data, tx_in_channels: int,
                       wallet_in_channels: int, config: Dict = None) -> Dict:
        """Test Heterogeneous GNN"""
        if config is None:
            config = {'hidden_channels': 128, 'num_layers': 2, 'dropout': 0.5}

        model = HeteroGNN(
            tx_in_channels=tx_in_channels,
            wallet_in_channels=wallet_in_channels,
            **config
        )

        results = self.train_model(model, data, is_hetero=True)
        results['model_name'] = 'HeteroGNN'
        results['config'] = config

        return results

    def test_hetero_gnn_attention(self, data, tx_in_channels: int,
                                  wallet_in_channels: int, config: Dict = None) -> Dict:
        """Test Heterogeneous GNN with Attention"""
        if config is None:
            config = {'hidden_channels': 128, 'num_layers': 2,
                     'heads': 4, 'dropout': 0.5}

        model = HeteroGNNWithAttention(
            tx_in_channels=tx_in_channels,
            wallet_in_channels=wallet_in_channels,
            **config
        )

        results = self.train_model(model, data, is_hetero=True)
        results['model_name'] = 'HeteroGNN_Attention'
        results['config'] = config

        return results

    def run_homogeneous_experiments(self, configs: List[Dict] = None) -> List[Dict]:
        """
        Run all homogeneous graph experiments (GCN, GAT, GraphSAGE)

        Args:
            configs: List of config dicts for each model, or None for defaults

        Returns:
            List of result dictionaries
        """
        data, in_channels = self._load_homogeneous_data()

        if configs is None:
            configs = [None, None, None]

        results = []

        print(f"Testing GCN (in_channels={in_channels})")
        gcn_results = self.test_gcn(data, in_channels, configs[0])
        results.append(gcn_results)
        self._print_results(gcn_results)

        print(f"\nTesting GAT (in_channels={in_channels})")
        gat_results = self.test_gat(data, in_channels, configs[1])
        results.append(gat_results)
        self._print_results(gat_results)

        print(f"\nTesting GraphSAGE (in_channels={in_channels})")
        sage_results = self.test_graphsage(data, in_channels, configs[2])
        results.append(sage_results)
        self._print_results(sage_results)

        return results

    def run_heterogeneous_experiments(self, configs: List[Dict] = None) -> List[Dict]:
        """
        Run heterogeneous graph experiments (HeteroGNN, HeteroGNN+Attention)

        Args:
            configs: List of config dicts for each model, or None for defaults

        Returns:
            List of result dictionaries
        """
        data, tx_in_channels, wallet_in_channels = self._load_heterogeneous_data()

        if configs is None:
            configs = [None, None]

        results = []

        print(f"Testing HeteroGNN (tx={tx_in_channels}, wallet={wallet_in_channels})")
        hetero_results = self.test_hetero_gnn(
            data, tx_in_channels, wallet_in_channels, configs[0]
        )
        results.append(hetero_results)
        self._print_results(hetero_results)

        print(f"\nTesting HeteroGNN+Attention (tx={tx_in_channels}, wallet={wallet_in_channels})")
        hetero_attn_results = self.test_hetero_gnn_attention(
            data, tx_in_channels, wallet_in_channels, configs[1]
        )
        results.append(hetero_attn_results)
        self._print_results(hetero_attn_results)

        return results

    def run_full_suite(self) -> pd.DataFrame:
        """
        Run all experiments and generate comparison report

        Returns:
            DataFrame with all results
        """
        all_results = []

        print("=" * 80)
        print("HOMOGENEOUS GRAPH EXPERIMENTS")
        print("=" * 80)
        homo_results = self.run_homogeneous_experiments()
        all_results.extend(homo_results)

        print("\n" + "=" * 80)
        print("HETEROGENEOUS GRAPH EXPERIMENTS")
        print("=" * 80)
        hetero_results = self.run_heterogeneous_experiments()
        all_results.extend(hetero_results)

        df = self._create_comparison_table(all_results)

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))

        self._save_results(all_results, df)

        return df

    def _print_results(self, results: Dict):
        """Print formatted results for a single model"""
        metrics = results['test_metrics']
        print(f"{results['model_name']}: F1={metrics['f1_illicit']:.4f}, "
              f"Acc={metrics['accuracy']:.4f}, "
              f"Prec={metrics['precision_illicit']:.4f}, "
              f"Rec={metrics['recall_illicit']:.4f}, "
              f"Time={results['training_time']:.2f}s")

    def _create_comparison_table(self, results: List[Dict]) -> pd.DataFrame:
        """Create comparison table from results"""
        rows = []

        for r in results:
            m = r['test_metrics']
            row = {
                'Model': r['model_name'],
                'F1_Illicit': m['f1_illicit'],
                'Accuracy': m['accuracy'],
                'Precision': m['precision_illicit'],
                'Recall': m['recall_illicit'],
                'F1_Macro': m['f1_macro'],
                'Training_Time': r['training_time'],
                'Epochs': r['final_epoch']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('F1_Illicit', ascending=False)

        return df

    def _save_results(self, results: List[Dict], df: pd.DataFrame):
        """Save results to disk"""
        output_file = self.results_dir / 'fraud_detection_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        df.to_csv(self.results_dir / 'fraud_detection_comparison.csv', index=False)

        with open(self.results_dir / 'fraud_detection_summary.txt', 'w') as f:
            f.write("FRAUD DETECTION MODEL COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            f.write(f"Best Model: {df.iloc[0]['Model']}\n")
            f.write(f"Best F1 Score: {df.iloc[0]['F1_Illicit']:.4f}\n")


def main():
    """Main entry point for test suite"""
    suite = FraudDetectionTestSuite(
        data_dir='.',
        results_dir='./results/fraud_detection',
        use_structural_features=False
    )

    results_df = suite.run_full_suite()

    return results_df


if __name__ == "__main__":
    results = main()
