#!/usr/bin/env python3
"""
Class imbalance handling comparison

Dataset: 4,545 illicit (9.76%) vs 42,019 licit (90.24%) - 1:9.25 ratio

Strategies tested:
1. Unweighted CE - No imbalance handling (baseline)
2. Weighted CE (alpha=0.5) - Moderate class weighting
3. Focal Loss (gamma=2.0) - Focus on hard examples
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from fraud_detection_suite import FraudDetectionTestSuite
from models import GraphSAGE
import pandas as pd


def run_strategy(suite, data, in_channels, name, loss_type, alpha, gamma):
    """Run single strategy and return results"""
    print(f"\nTesting: {name}")
    print("-" * 60)

    model = GraphSAGE(in_channels=in_channels, hidden_channels=64, num_layers=2)

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
        'Strategy': name,
        'Loss': loss_type,
        'Alpha': alpha,
        'Gamma': gamma,
        'F1': m['f1_illicit'],
        'Precision': m['precision_illicit'],
        'Recall': m['recall_illicit'],
        'Accuracy': m['accuracy'],
        'Time': result['training_time']
    }


def main():
    print("=" * 70)
    print("CLASS IMBALANCE HANDLING EXPERIMENTS")
    print("=" * 70)
    print("Dataset: 9.76% illicit vs 90.24% licit (1:9.25 ratio)")
    print("Model: GraphSAGE (64 hidden, 2 layers)")
    print("=" * 70)

    suite = FraudDetectionTestSuite()
    data, in_channels = suite._load_homogeneous_data()

    strategies = [
        ("Baseline (No Weighting)", "ce_unweighted", 0.0, 0.0),
        ("Weighted CE (alpha=0.5)", "ce", 0.5, 0.0),
        ("Weighted CE (alpha=1.0)", "ce", 1.0, 0.0),
        ("Focal Loss (gamma=2.0, alpha=0.5)", "focal", 0.5, 2.0),
        ("Focal Loss (gamma=2.0, alpha=1.0)", "focal", 1.0, 2.0),
    ]

    results = []
    for name, loss_type, alpha, gamma in strategies:
        results.append(run_strategy(suite, data, in_channels, name, loss_type, alpha, gamma))

    df = pd.DataFrame(results).sort_values('F1', ascending=False)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    df.to_csv('results/fraud_detection/imbalance_comparison.csv', index=False)
    print("\nSaved: results/fraud_detection/imbalance_comparison.csv")

    best = df.iloc[0]
    print(f"\nBest: {best['Strategy']} (F1={best['F1']:.4f})")


if __name__ == "__main__":
    main()
