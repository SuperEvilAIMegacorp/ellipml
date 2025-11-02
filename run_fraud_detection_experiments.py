#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from fraud_detection_suite import FraudDetectionTestSuite
from experiment_config import get_experiment_suite, get_training_config, get_model_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run fraud detection experiments on Bitcoin transaction graphs'
    )

    parser.add_argument(
        '--suite',
        type=str,
        default='baseline_comparison',
        choices=['quick_test', 'baseline_comparison', 'hetero_test',
                'full_comparison', 'architecture_search'],
        help='Predefined experiment suite to run'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['gcn', 'gat', 'sage', 'hetero', 'hetero_attn'],
        help='Specific models to test (overrides suite)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory containing elliptic and ellipticpp data'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results/fraud_detection',
        help='Directory to save results'
    )

    parser.add_argument(
        '--use-structural-features',
        action='store_true',
        help='Augment features with structural graph features'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum training epochs'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Hidden dimension for models'
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of GNN layers'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate'
    )

    return parser.parse_args()


def run_suite(args):
    """Run predefined experiment suite"""
    suite = FraudDetectionTestSuite(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=args.device,
        use_structural_features=args.use_structural_features
    )

    print(f"Running experiment suite: {args.suite}")
    results_df = suite.run_full_suite()

    return results_df


def run_custom_models(args):
    """Run specific models with custom configuration"""
    suite = FraudDetectionTestSuite(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=args.device,
        use_structural_features=args.use_structural_features
    )

    custom_config = {
        'hidden_channels': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }

    all_results = []

    if any(m in args.models for m in ['gcn', 'gat', 'sage']):
        data, in_channels = suite._load_homogeneous_data()

        if 'gcn' in args.models:
            print(f"\nTesting GCN")
            results = suite.test_gcn(data, in_channels, custom_config)
            all_results.append(results)
            suite._print_results(results)

        if 'gat' in args.models:
            print(f"\nTesting GAT")
            gat_config = {**custom_config, 'heads': 4}
            results = suite.test_gat(data, in_channels, gat_config)
            all_results.append(results)
            suite._print_results(results)

        if 'sage' in args.models:
            print(f"\nTesting GraphSAGE")
            results = suite.test_graphsage(data, in_channels, custom_config)
            all_results.append(results)
            suite._print_results(results)

    if any(m in args.models for m in ['hetero', 'hetero_attn']):
        data, tx_in_channels, wallet_in_channels = suite._load_heterogeneous_data()

        if 'hetero' in args.models:
            print(f"\nTesting HeteroGNN")
            results = suite.test_hetero_gnn(
                data, tx_in_channels, wallet_in_channels, custom_config
            )
            all_results.append(results)
            suite._print_results(results)

        if 'hetero_attn' in args.models:
            print(f"\nTesting HeteroGNN+Attention")
            hetero_attn_config = {**custom_config, 'heads': 4}
            results = suite.test_hetero_gnn_attention(
                data, tx_in_channels, wallet_in_channels, hetero_attn_config
            )
            all_results.append(results)
            suite._print_results(results)

    df = suite._create_comparison_table(all_results)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    suite._save_results(all_results, df)

    return df


def main():
    args = parse_args()

    print("=" * 80)
    print("FRAUD DETECTION MODEL TEST SUITE")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Structural features: {args.use_structural_features}")
    print("=" * 80)

    if args.models:
        results_df = run_custom_models(args)
    else:
        results_df = run_suite(args)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    best_model = results_df.iloc[0]
    print(f"\nBest performing model: {best_model['Model']}")
    print(f"F1 Score (Illicit): {best_model['F1_Illicit']:.4f}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"Precision: {best_model['Precision']:.4f}")
    print(f"Recall: {best_model['Recall']:.4f}")


if __name__ == "__main__":
    main()
