#!/usr/bin/env python3
"""
Comprehensive imbalance handling comparison across all models

Tests all 5 models (GCN, GAT, GraphSAGE, HeteroGNN, HeteroGNN+Attention)
with all 5 imbalance strategies for systematic comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from fraud_detection_suite import FraudDetectionTestSuite
from models import GCN, GAT, GraphSAGE, HeteroGNN
import pandas as pd


STRATEGIES = [
    ("Baseline", "ce_unweighted", 0.0, 0.0),
    ("Weighted CE (α=0.5)", "ce", 0.5, 0.0),
    ("Weighted CE (α=1.0)", "ce", 1.0, 0.0),
    ("Focal Loss (α=0.5, γ=2.0)", "focal", 0.5, 2.0),
    ("Focal Loss (α=1.0, γ=2.0)", "focal", 1.0, 2.0),
]


def run_homogeneous_model(suite, data, in_channels, model_class, model_name,
                          strategy_name, loss_type, alpha, gamma):
    """Run single homogeneous model with given strategy"""
    print(f"\n{model_name} + {strategy_name}")
    print("-" * 60)

    model = model_class(in_channels=in_channels, hidden_channels=64, num_layers=2)

    result = suite.train_model(
        model, data,
        is_hetero=False,
        epochs=100,
        loss_type=loss_type,
        class_weight_alpha=alpha,
        focal_gamma=gamma
    )

    m = result['test_metrics']
    print(f"F1={m['f1_illicit']:.4f}, Prec={m['precision_illicit']:.4f}, "
          f"Rec={m['recall_illicit']:.4f}, Time={result['training_time']:.2f}s")

    return {
        'Model': model_name,
        'Graph_Type': 'Homogeneous',
        'Strategy': strategy_name,
        'Loss_Type': loss_type,
        'Alpha': alpha,
        'Gamma': gamma,
        'F1': m['f1_illicit'],
        'Precision': m['precision_illicit'],
        'Recall': m['recall_illicit'],
        'Accuracy': m['accuracy'],
        'Time': result['training_time']
    }


def run_heterogeneous_model(suite, data, tx_in, wallet_in, model_class, model_name,
                            strategy_name, loss_type, alpha, gamma, use_attention=False):
    """Run single heterogeneous model with given strategy"""
    print(f"\n{model_name} + {strategy_name}")
    print("-" * 60)

    if use_attention:
        model = model_class(
            tx_in_channels=tx_in,
            wallet_in_channels=wallet_in,
            hidden_channels=64,
            num_layers=2,
            heads=4
        )
    else:
        model = model_class(
            tx_in_channels=tx_in,
            wallet_in_channels=wallet_in,
            hidden_channels=64,
            num_layers=2
        )

    result = suite.train_model(
        model, data,
        is_hetero=True,
        epochs=100,
        loss_type=loss_type,
        class_weight_alpha=alpha,
        focal_gamma=gamma
    )

    m = result['test_metrics']
    print(f"F1={m['f1_illicit']:.4f}, Prec={m['precision_illicit']:.4f}, "
          f"Rec={m['recall_illicit']:.4f}, Time={result['training_time']:.2f}s")

    return {
        'Model': model_name,
        'Graph_Type': 'Heterogeneous',
        'Strategy': strategy_name,
        'Loss_Type': loss_type,
        'Alpha': alpha,
        'Gamma': gamma,
        'F1': m['f1_illicit'],
        'Precision': m['precision_illicit'],
        'Recall': m['recall_illicit'],
        'Accuracy': m['accuracy'],
        'Time': result['training_time']
    }


def main():
    print("=" * 80)
    print("COMPREHENSIVE IMBALANCE HANDLING COMPARISON")
    print("=" * 80)
    print("Models: GCN, GAT, GraphSAGE, HeteroGNN")
    print("Strategies: 5 different imbalance handling approaches")
    print("Total experiments: 20 (4 models × 5 strategies)")
    print("Note: HeteroGNN+Attention excluded (OOM + poor performance)")
    print("=" * 80)

    suite = FraudDetectionTestSuite()

    homo_data, in_channels = suite._load_homogeneous_data()
    hetero_data, tx_in, wallet_in = suite._load_heterogeneous_data()

    results = []

    for strategy_name, loss_type, alpha, gamma in STRATEGIES:
        print(f"\n\n{'=' * 80}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'=' * 80}")

        results.append(run_homogeneous_model(
            suite, homo_data, in_channels, GCN, "GCN",
            strategy_name, loss_type, alpha, gamma
        ))

        results.append(run_homogeneous_model(
            suite, homo_data, in_channels, GAT, "GAT",
            strategy_name, loss_type, alpha, gamma
        ))

        results.append(run_homogeneous_model(
            suite, homo_data, in_channels, GraphSAGE, "GraphSAGE",
            strategy_name, loss_type, alpha, gamma
        ))

        results.append(run_heterogeneous_model(
            suite, hetero_data, tx_in, wallet_in, HeteroGNN, "HeteroGNN",
            strategy_name, loss_type, alpha, gamma
        ))

    df = pd.DataFrame(results)

    print("\n\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    df_sorted = df.sort_values('F1', ascending=False)

    print("\n\n" + "=" * 80)
    print("TOP 10 COMBINATIONS (by F1 Score)")
    print("=" * 80)
    print(df_sorted.head(10).to_string(index=False))

    output_file = 'results/fraud_detection/full_imbalance_comparison.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"\n\nSaved: {output_file}")

    best = df_sorted.iloc[0]
    print(f"\nBest combination:")
    print(f"  {best['Model']} + {best['Strategy']}")
    print(f"  F1={best['F1']:.4f}, Precision={best['Precision']:.4f}, Recall={best['Recall']:.4f}")


if __name__ == "__main__":
    main()
