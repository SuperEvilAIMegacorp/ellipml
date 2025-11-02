from typing import Dict, List


BASELINE_CONFIGS = {
    'gcn_small': {
        'hidden_channels': 64,
        'num_layers': 2,
        'dropout': 0.5
    },
    'gcn_large': {
        'hidden_channels': 256,
        'num_layers': 3,
        'dropout': 0.5
    },
    'gat_small': {
        'hidden_channels': 64,
        'num_layers': 2,
        'heads': 4,
        'dropout': 0.5
    },
    'gat_large': {
        'hidden_channels': 128,
        'num_layers': 3,
        'heads': 8,
        'dropout': 0.5
    },
    'sage_small': {
        'hidden_channels': 64,
        'num_layers': 2,
        'dropout': 0.5
    },
    'sage_large': {
        'hidden_channels': 256,
        'num_layers': 3,
        'dropout': 0.5
    },
    'hetero_small': {
        'hidden_channels': 64,
        'num_layers': 2,
        'dropout': 0.5
    },
    'hetero_large': {
        'hidden_channels': 256,
        'num_layers': 3,
        'dropout': 0.5
    },
    'hetero_attn_small': {
        'hidden_channels': 64,
        'num_layers': 2,
        'heads': 4,
        'dropout': 0.5
    },
    'hetero_attn_large': {
        'hidden_channels': 128,
        'num_layers': 3,
        'heads': 8,
        'dropout': 0.5
    }
}


TRAINING_CONFIGS = {
    'fast': {
        'epochs': 50,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'patience': 10
    },
    'standard': {
        'epochs': 100,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'patience': 20
    },
    'extensive': {
        'epochs': 200,
        'lr': 0.005,
        'weight_decay': 5e-4,
        'patience': 30
    }
}


EXPERIMENT_SUITES = {
    'quick_test': {
        'description': 'Fast test of all models with small configs',
        'models': ['gcn', 'gat', 'sage'],
        'model_configs': ['small'] * 3,
        'training_config': 'fast'
    },
    'baseline_comparison': {
        'description': 'Compare all baseline models with standard configs',
        'models': ['gcn', 'gat', 'sage'],
        'model_configs': ['small'] * 3,
        'training_config': 'standard'
    },
    'hetero_test': {
        'description': 'Test heterogeneous models',
        'models': ['hetero', 'hetero_attn'],
        'model_configs': ['small', 'small'],
        'training_config': 'standard'
    },
    'full_comparison': {
        'description': 'Comprehensive comparison of all models',
        'models': ['gcn', 'gat', 'sage', 'hetero', 'hetero_attn'],
        'model_configs': ['small'] * 5,
        'training_config': 'standard'
    },
    'architecture_search': {
        'description': 'Test different architectures for each model type',
        'models': ['gcn', 'gcn', 'gat', 'gat', 'sage', 'sage'],
        'model_configs': ['small', 'large', 'small', 'large', 'small', 'large'],
        'training_config': 'extensive'
    }
}


def get_model_config(model_name: str, size: str = 'small') -> Dict:
    """Get configuration for a specific model"""
    config_key = f"{model_name}_{size}"
    if config_key not in BASELINE_CONFIGS:
        raise ValueError(f"Unknown config: {config_key}")
    return BASELINE_CONFIGS[config_key]


def get_training_config(config_name: str = 'standard') -> Dict:
    """Get training configuration"""
    if config_name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown training config: {config_name}")
    return TRAINING_CONFIGS[config_name]


def get_experiment_suite(suite_name: str) -> Dict:
    """Get predefined experiment suite"""
    if suite_name not in EXPERIMENT_SUITES:
        raise ValueError(f"Unknown experiment suite: {suite_name}")
    return EXPERIMENT_SUITES[suite_name]
