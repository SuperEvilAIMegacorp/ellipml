#!/usr/bin/env python3
"""
Validate fraud pattern detection on labeled data

Analyzes whether peel chains, mixing services, and rapid dispersal
patterns are more prevalent in illicit vs licit transactions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from data_loader import EllipticDataLoader
from fraud_patterns import FraudPatternDetector, analyze_pattern_distribution


def main():
    print("=" * 70)
    print("FRAUD PATTERN VALIDATION")
    print("=" * 70)

    loader = EllipticDataLoader()
    data = loader.prepare_graph_data(use_fraud_patterns=False)

    print("\n" + "=" * 70)
    print("COMPUTING PATTERN SCORES")
    print("=" * 70)

    tx_features, tx_classes, _ = loader.load_raw_data()
    tx_data = tx_features.merge(tx_classes, on='txId', how='left')
    timestamps = tx_data['Time step'].values

    detector = FraudPatternDetector(
        data.edge_index,
        data.x.size(0),
        timestamps
    )

    patterns = detector.compute_all_patterns()

    print("\n" + "=" * 70)
    print("PATTERN SCORE STATISTICS")
    print("=" * 70)

    labels = data.y.numpy()
    labeled_mask = (labels != 2)

    stats = analyze_pattern_distribution(patterns, labels, labeled_mask)

    results = []
    for pattern_name, pattern_stats in stats.items():
        print(f"\n{pattern_name}:")
        print(f"  Illicit: mean={pattern_stats['illicit_mean']:.4f}, "
              f"median={pattern_stats['illicit_median']:.4f}, "
              f"std={pattern_stats['illicit_std']:.4f}")
        print(f"  Licit:   mean={pattern_stats['licit_mean']:.4f}, "
              f"median={pattern_stats['licit_median']:.4f}, "
              f"std={pattern_stats['licit_std']:.4f}")
        print(f"  Separation: {pattern_stats['separation']:.4f}")

        results.append({
            'Pattern': pattern_name,
            'Illicit_Mean': pattern_stats['illicit_mean'],
            'Licit_Mean': pattern_stats['licit_mean'],
            'Separation': pattern_stats['separation']
        })

    print("\n" + "=" * 70)
    print("HIGH-SCORING NODES ANALYSIS")
    print("=" * 70)

    for pattern_name, scores in patterns.items():
        print(f"\n{pattern_name}:")

        threshold = np.percentile(scores[scores > 0], 90) if (scores > 0).any() else 0
        high_score_mask = (scores > threshold) & labeled_mask

        if high_score_mask.sum() > 0:
            high_score_labels = labels[high_score_mask]
            illicit_pct = (high_score_labels == 0).sum() / len(high_score_labels) * 100

            print(f"  Nodes with score > {threshold:.3f}: {high_score_mask.sum()}")
            print(f"  Illicit rate in high-scoring: {illicit_pct:.1f}%")
            print(f"  Baseline illicit rate: {(labels[labeled_mask] == 0).sum() / labeled_mask.sum() * 100:.1f}%")
            print(f"  Enrichment: {illicit_pct / ((labels[labeled_mask] == 0).sum() / labeled_mask.sum() * 100):.2f}x")
        else:
            print("  No high-scoring nodes found")

    print("\n" + "=" * 70)
    print("PATTERN CO-OCCURRENCE")
    print("=" * 70)

    peel_high = patterns['peel_chain_score'] > np.percentile(patterns['peel_chain_score'][patterns['peel_chain_score'] > 0], 75) if (patterns['peel_chain_score'] > 0).any() else False
    mixing_high = patterns['mixing_service_score'] > np.percentile(patterns['mixing_service_score'][patterns['mixing_service_score'] > 0], 75) if (patterns['mixing_service_score'] > 0).any() else False
    dispersal_high = patterns['rapid_dispersal_score'] > np.percentile(patterns['rapid_dispersal_score'][patterns['rapid_dispersal_score'] > 0], 75) if (patterns['rapid_dispersal_score'] > 0).any() else False

    peel_mixing = (peel_high & mixing_high & labeled_mask).sum()
    peel_dispersal = (peel_high & dispersal_high & labeled_mask).sum()
    mixing_dispersal = (mixing_high & dispersal_high & labeled_mask).sum()
    all_three = (peel_high & mixing_high & dispersal_high & labeled_mask).sum()

    print(f"Peel + Mixing: {peel_mixing} nodes")
    print(f"Peel + Dispersal: {peel_dispersal} nodes")
    print(f"Mixing + Dispersal: {mixing_dispersal} nodes")
    print(f"All three patterns: {all_three} nodes")

    df = pd.DataFrame(results)
    output_file = 'results/fraud_detection/pattern_validation.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total_separation = sum(s['separation'] for s in stats.values())
    if total_separation > 0.1:
        print("Patterns show positive separation (illicit > licit)")
        print("Recommendation: Use fraud patterns for training")
    elif total_separation < -0.1:
        print("Warning: Patterns show negative separation")
        print("Recommendation: Do not use these patterns")
    else:
        print("Patterns show minimal separation")
        print("Recommendation: Patterns may not improve performance")


if __name__ == "__main__":
    main()
