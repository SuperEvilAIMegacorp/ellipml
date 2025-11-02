#!/usr/bin/env python3
"""
Comparative experiments: Baseline vs Fraud Pattern Features

Tests whether fraud pattern features (peel chains, mixing services,
rapid dispersal) improve fraud detection performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from fraud_detection_suite import FraudDetectionTestSuite
from models import GCN, GAT, GraphSAGE, HeteroGNN
import pandas as pd


def run_model(suite, model_class, model_name, data, in_channels, use_patterns):
    """Run single model configuration"""
    feature_type = "Patterns" if use_patterns else "Baseline"
    print(f"\n{model_name} + {feature_type}")
    print("-" * 60)

    if model_name == "HeteroGNN":
        model = model_class(
            tx_in_channels=in_channels[0],
            wallet_in_channels=in_channels[1],
            hidden_channels=64,
            num_layers=2
        )
        is_hetero = True
    else:
        model = model_class(
            in_channels=in_channels,
            hidden_channels=64,
            num_layers=2
        )
        is_hetero = False

    result = suite.train_model(
        model, data,
        is_hetero=is_hetero,
        epochs=100,
        loss_type='ce_unweighted'
    )

    m = result['test_metrics']
    print(f"F1={m['f1_illicit']:.4f}, Prec={m['precision_illicit']:.4f}, "
          f"Rec={m['recall_illicit']:.4f}, Time={result['training_time']:.2f}s")

    return {
        'Model': model_name,
        'Features': feature_type,
        'F1': m['f1_illicit'],
        'Precision': m['precision_illicit'],
        'Recall': m['recall_illicit'],
        'Accuracy': m['accuracy'],
        'Time': result['training_time'],
        'Input_Dim': in_channels if not is_hetero else in_channels[0]
    }


def main():
    print("=" * 70)
    print("FRAUD PATTERN FEATURE COMPARISON")
    print("=" * 70)
    print("Comparing: Baseline (183 features) vs Patterns (186 features)")
    print("Models: GCN, GAT, GraphSAGE, HeteroGNN")
    print("=" * 70)

    results = []

    print("\n" + "=" * 70)
    print("BASELINE FEATURES (183 original features)")
    print("=" * 70)

    suite_baseline = FraudDetectionTestSuite(use_fraud_patterns=False)
    homo_data, in_channels = suite_baseline._load_homogeneous_data()
    hetero_data, tx_in, wallet_in = suite_baseline._load_heterogeneous_data()

    results.append(run_model(suite_baseline, GCN, "GCN", homo_data, in_channels, False))
    results.append(run_model(suite_baseline, GAT, "GAT", homo_data, in_channels, False))
    results.append(run_model(suite_baseline, GraphSAGE, "GraphSAGE", homo_data, in_channels, False))
    results.append(run_model(suite_baseline, HeteroGNN, "HeteroGNN", hetero_data, (tx_in, wallet_in), False))

    print("\n" + "=" * 70)
    print("PATTERN FEATURES (183 + 3 fraud patterns = 186 features)")
    print("=" * 70)

    suite_patterns = FraudDetectionTestSuite(use_fraud_patterns=True)
    homo_data_p, in_channels_p = suite_patterns._load_homogeneous_data()
    hetero_data_p, tx_in_p, wallet_in_p = suite_patterns._load_heterogeneous_data()

    results.append(run_model(suite_patterns, GCN, "GCN", homo_data_p, in_channels_p, True))
    results.append(run_model(suite_patterns, GAT, "GAT", homo_data_p, in_channels_p, True))
    results.append(run_model(suite_patterns, GraphSAGE, "GraphSAGE", homo_data_p, in_channels_p, True))
    results.append(run_model(suite_patterns, HeteroGNN, "HeteroGNN", hetero_data_p, (tx_in_p, wallet_in_p), True))

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("PERFORMANCE GAIN ANALYSIS")
    print("=" * 70)

    for model_name in ["GCN", "GAT", "GraphSAGE", "HeteroGNN"]:
        baseline_row = df[(df['Model'] == model_name) & (df['Features'] == 'Baseline')].iloc[0]
        pattern_row = df[(df['Model'] == model_name) & (df['Features'] == 'Patterns')].iloc[0]

        f1_gain = (pattern_row['F1'] - baseline_row['F1']) * 100
        prec_gain = (pattern_row['Precision'] - baseline_row['Precision']) * 100
        rec_gain = (pattern_row['Recall'] - baseline_row['Recall']) * 100

        print(f"\n{model_name}:")
        print(f"  F1 change: {f1_gain:+.2f}%")
        print(f"  Precision change: {prec_gain:+.2f}%")
        print(f"  Recall change: {rec_gain:+.2f}%")

        if f1_gain > 2.0:
            print(f"  Status: Significant improvement")
        elif f1_gain > 0.5:
            print(f"  Status: Marginal improvement")
        elif f1_gain > -0.5:
            print(f"  Status: No change")
        else:
            print(f"  Status: Degradation")

    output_file = 'results/fraud_detection/pattern_comparison.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    best_baseline = df[df['Features'] == 'Baseline'].sort_values('F1', ascending=False).iloc[0]
    best_pattern = df[df['Features'] == 'Patterns'].sort_values('F1', ascending=False).iloc[0]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)
    print(f"Best Baseline: {best_baseline['Model']} (F1={best_baseline['F1']:.4f})")
    print(f"Best Pattern:  {best_pattern['Model']} (F1={best_pattern['F1']:.4f})")

    if best_pattern['F1'] > best_baseline['F1']:
        improvement = (best_pattern['F1'] - best_baseline['F1']) * 100
        print(f"\nPattern features improve best model by {improvement:.2f}%")
        print("Recommendation: Use fraud pattern features")
    else:
        decline = (best_baseline['F1'] - best_pattern['F1']) * 100
        print(f"\nPattern features degrade best model by {decline:.2f}%")
        print("Recommendation: Use baseline features only")


if __name__ == "__main__":
    main()
