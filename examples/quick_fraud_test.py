import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fraud_detection_suite import FraudDetectionTestSuite


def quick_baseline_test():
    """
    Quick test comparing baseline models (GCN, GAT, GraphSAGE)
    """
    suite = FraudDetectionTestSuite(
        data_dir='..',
        results_dir='../results/fraud_detection',
        use_structural_features=False
    )

    print("Running baseline model comparison...")
    results = suite.run_homogeneous_experiments()

    print("\nResults:")
    for r in results:
        m = r['test_metrics']
        print(f"{r['model_name']:15} | F1: {m['f1_illicit']:.4f} | "
              f"Acc: {m['accuracy']:.4f} | Time: {r['training_time']:.1f}s")


def test_heterogeneous_models():
    """
    Test heterogeneous models that use both transaction and wallet data
    """
    suite = FraudDetectionTestSuite(
        data_dir='..',
        results_dir='../results/fraud_detection',
        use_structural_features=False
    )

    print("Running heterogeneous model comparison...")
    results = suite.run_heterogeneous_experiments()

    print("\nResults:")
    for r in results:
        m = r['test_metrics']
        print(f"{r['model_name']:20} | F1: {m['f1_illicit']:.4f} | "
              f"Acc: {m['accuracy']:.4f} | Time: {r['training_time']:.1f}s")


def full_comparison():
    """
    Run complete test suite with all models
    """
    suite = FraudDetectionTestSuite(
        data_dir='..',
        results_dir='../results/fraud_detection',
        use_structural_features=False
    )

    print("Running full model comparison...")
    results_df = suite.run_full_suite()

    print("\nFinal Rankings:")
    print(results_df[['Model', 'F1_Illicit', 'Accuracy', 'Training_Time']])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'hetero', 'full'],
                       default='baseline')
    args = parser.parse_args()

    if args.mode == 'baseline':
        quick_baseline_test()
    elif args.mode == 'hetero':
        test_heterogeneous_models()
    else:
        full_comparison()
