# Bitcoin Fraud Detection with Graph Neural Networks

A test suite for evaluating Graph Neural Network architectures on Bitcoin fraud detection using the Elliptic++ dataset.

## Installation

```bash
pip install -r requirements.txt
python test_setup.py
```

Requirements: Python 3.8+, PyTorch 2.0+, PyTorch Geometric 2.3+

## Usage

### Basic Commands

```bash
# Run baseline models
python run_fraud_detection_experiments.py --suite baseline_comparison

# Test specific models
python run_fraud_detection_experiments.py --models gcn gat sage

# Custom configuration
python run_fraud_detection_experiments.py --models hetero --epochs 100 --hidden-dim 128
```

### Programmatic API

```python
from src.fraud_detection_suite import FraudDetectionTestSuite

suite = FraudDetectionTestSuite()
results = suite.run_full_suite()
```

## Command Line Options

```
--suite NAME              Experiment suite (quick_test, baseline_comparison, full_comparison)
--models MODEL [...]      Models to test (gcn, gat, sage, hetero)
--data-dir PATH           Dataset directory (default: .)
--results-dir PATH        Output directory (default: ./results/fraud_detection)
--epochs N                Training epochs (default: 100)
--hidden-dim N            Hidden layer size (default: 128)
--num-layers N            Number of layers (default: 2)
--lr FLOAT                Learning rate (default: 0.01)
--dropout FLOAT           Dropout rate (default: 0.5)
--device DEVICE           cpu or cuda (auto-detect)
--use-structural-features Enable graph structure features
```

## Available Models

**Homogeneous (Transaction Graph)**
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network
- GraphSAGE: Sampling-based GNN

**Heterogeneous (Transaction + Wallet Graphs)**
- HeteroGNN: Multi-relation message passing
- HeteroGNN+Attention: Attention-enhanced

## Repository Structure

```
src/                    Core implementation
├── data_loader.py      Dataset loading
├── models.py           GNN architectures
├── train_utils.py      Training/evaluation
├── fraud_patterns.py   Pattern detection
└── fraud_detection_suite.py

ellipml/                Structural analysis
├── structure_discovery.py
├── tagging_engine.py
└── run_analysis.py

results/                Experiment outputs (JSON/CSV/PNG)
examples/               Usage examples
ellipticpp/             Dataset files (DOWNLOAD FROM GOOGLE DRIVE)
```

## Dataset

Elliptic++ (KDD 2023):
- 203,769 transactions (183 features)
- 822,942 wallets (8 features)
- 4.4M edges
- 4,545 illicit / 42,019 licit labels


## Output Files

Results saved to `results/fraud_detection/`:
- `*.json` - Detailed metrics
- `*.csv` - Tabular results
- `*.txt` - Summary
- `visualizations/*.png` - Graphs

## Additional Tools

```bash
# Structural analysis
cd ellipml && python run_analysis.py

# Class imbalance experiments
python run_imbalance_experiments.py

# Pattern feature experiments
python run_pattern_experiments.py

# Generate visualizations
python generate_visualizations.py
```
## Testing

```bash
python test_setup.py
python examples/quick_fraud_test.py
```

