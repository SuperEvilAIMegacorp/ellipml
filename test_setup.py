#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch


def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")

    try:
        from data_loader import EllipticDataLoader, EllipticPlusPlusDataLoader
        print("  Data loaders: OK")
    except Exception as e:
        print(f"  Data loaders: FAILED - {e}")
        return False

    try:
        from models import GCN, GAT, GraphSAGE, HeteroGNN, HeteroGNNWithAttention
        print("  Models: OK")
    except Exception as e:
        print(f"  Models: FAILED - {e}")
        return False

    try:
        from train_utils import train_epoch, evaluate
        print("  Training utilities: OK")
    except Exception as e:
        print(f"  Training utilities: FAILED - {e}")
        return False

    try:
        from structural_features import StructuralFeatureEngineer
        print("  Structural features: OK")
    except Exception as e:
        print(f"  Structural features: FAILED - {e}")
        return False

    try:
        from fraud_detection_suite import FraudDetectionTestSuite
        print("  Test suite: OK")
    except Exception as e:
        print(f"  Test suite: FAILED - {e}")
        return False

    return True


def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")

    try:
        from data_loader import EllipticDataLoader
        loader = EllipticDataLoader(data_dir='.')

        print("  Loading transaction data...")
        data = loader.prepare_graph_data()

        print(f"  Nodes: {data.x.size(0)}")
        print(f"  Features: {data.x.size(1)}")
        print(f"  Edges: {data.edge_index.size(1)}")
        print(f"  Train samples: {data.train_mask.sum()}")
        print(f"  Val samples: {data.val_mask.sum()}")
        print(f"  Test samples: {data.test_mask.sum()}")

        return True
    except Exception as e:
        print(f"  Data loading FAILED: {e}")
        return False


def test_model_creation():
    """Test model instantiation"""
    print("\nTesting model creation...")

    try:
        from models import GCN, GAT, GraphSAGE

        gcn = GCN(in_channels=166, hidden_channels=64, num_classes=2)
        print(f"  GCN: {sum(p.numel() for p in gcn.parameters()):,} parameters")

        gat = GAT(in_channels=166, hidden_channels=64, num_classes=2)
        print(f"  GAT: {sum(p.numel() for p in gat.parameters()):,} parameters")

        sage = GraphSAGE(in_channels=166, hidden_channels=64, num_classes=2)
        print(f"  GraphSAGE: {sum(p.numel() for p in sage.parameters()):,} parameters")

        return True
    except Exception as e:
        print(f"  Model creation FAILED: {e}")
        return False


def test_device():
    """Test GPU availability"""
    print("\nTesting compute device...")

    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  CUDA not available, using CPU")

    return True


def main():
    print("=" * 60)
    print("FRAUD DETECTION TEST SUITE - SETUP VERIFICATION")
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_device()
    all_passed &= test_data_loading()
    all_passed &= test_model_creation()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - Setup is ready!")
    else:
        print("SOME TESTS FAILED - Check errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
