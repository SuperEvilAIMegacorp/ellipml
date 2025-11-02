#!/usr/bin/env python3
"""
Generate comprehensive graph visualizations for fraud detection analysis

Produces 5 publication-quality visualizations:
1. Confusion subgraphs (TP, FP, FN, TN)
2. Peel chain patterns
3. Degree distribution with fraud rate
4. Prediction confidence heatmap
5. Heterogeneous graph structure
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from fraud_detection_suite import FraudDetectionTestSuite
from models import HeteroGNN
from graph_visualizations import FraudGraphVisualizer
from fraud_patterns import FraudPatternDetector


def train_model_for_predictions(suite, data, in_channels, is_hetero=False):
    """Train model to get predictions for visualization"""
    print("Training model for predictions...")

    if is_hetero:
        model = HeteroGNN(
            tx_in_channels=in_channels[0],
            wallet_in_channels=in_channels[1],
            hidden_channels=64,
            num_layers=2
        )
    else:
        from models import GraphSAGE
        model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=64,
            num_layers=2
        )

    result = suite.train_model(
        model, data,
        is_hetero=is_hetero,
        epochs=100,
        loss_type='ce_unweighted'
    )

    model.eval()
    with torch.no_grad():
        if is_hetero:
            out_dict = model(data.x_dict, data.edge_index_dict)
            out = out_dict['transaction']
        else:
            out = model(data.x, data.edge_index)

        predictions = out.argmax(dim=1).cpu().numpy()

    m = result['test_metrics']
    print(f"Model trained: F1={m['f1_illicit']:.4f}, Prec={m['precision_illicit']:.4f}, "
          f"Rec={m['recall_illicit']:.4f}")

    return predictions


def compute_pattern_scores(data):
    """Compute fraud pattern scores for visualization"""
    print("Computing fraud pattern scores...")

    from data_loader import EllipticDataLoader
    loader = EllipticDataLoader()
    tx_features, tx_classes, _ = loader.load_raw_data()
    tx_data = tx_features.merge(tx_classes, on='txId', how='left')
    timestamps = tx_data['Time step'].values

    detector = FraudPatternDetector(
        data.edge_index,
        data.x.size(0),
        timestamps
    )

    patterns = detector.compute_all_patterns()
    print("Pattern scores computed")

    return patterns


def main():
    print("=" * 70)
    print("GRAPH VISUALIZATION GENERATION")
    print("=" * 70)

    output_dir = Path("results/fraud_detection/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    suite_homo = FraudDetectionTestSuite(use_fraud_patterns=True)
    homo_data, in_channels = suite_homo._load_homogeneous_data()

    suite_hetero = FraudDetectionTestSuite(use_fraud_patterns=False)
    hetero_data, tx_in, wallet_in = suite_hetero._load_heterogeneous_data()

    pattern_scores = compute_pattern_scores(homo_data)

    print("\n" + "=" * 70)
    print("TRAINING MODEL FOR PREDICTIONS")
    print("=" * 70)

    predictions = train_model_for_predictions(
        suite_hetero,
        hetero_data,
        (tx_in, wallet_in),
        is_hetero=True
    )

    tx_data_for_viz = homo_data
    tx_data_for_viz.predictions = predictions

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualizer = FraudGraphVisualizer(
        tx_data_for_viz,
        predictions=predictions,
        pattern_scores=pattern_scores
    )

    print("\n[1/5] Confusion Subgraphs...")
    visualizer.plot_confusion_subgraphs(output_dir, max_nodes_per_type=30)

    print("\n[2/5] Peel Chain Patterns...")
    visualizer.plot_peel_chain_subgraphs(output_dir, num_chains=6)

    print("\n[3/5] Degree Distribution...")
    visualizer.plot_degree_distribution(output_dir)

    print("\n[4/5] Prediction Heatmap...")
    visualizer.plot_prediction_heatmap(output_dir, sample_size=500)

    print("\n[5/5] Heterogeneous Graph Structure...")
    visualizer.plot_heterogeneous_structure(output_dir, hetero_data=hetero_data, sample_txs=200)

    print("\n" + "=" * 70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated 5 visualizations in: {output_dir}")
    print("\nFiles created:")
    print("  1. confusion_subgraphs.png - Error analysis (TP/FP/FN/TN)")
    print("  2. peel_chain_patterns.png - Money laundering patterns")
    print("  3. degree_distribution.png - Degree vs fraud rate")
    print("  4. prediction_heatmap.png - Model confidence visualization")
    print("  5. heterogeneous_structure.png - Transaction-wallet graph")


if __name__ == "__main__":
    main()
