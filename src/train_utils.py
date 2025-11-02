"""
Training and evaluation utilities
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
import numpy as np
from typing import Dict, Tuple


def train_epoch(model, data, optimizer, device, is_hetero=False, loss_fn=None):
    """
    Train for one epoch

    Args:
        model: PyTorch model
        data: PyG Data or HeteroData object
        optimizer: PyTorch optimizer
        device: torch device
        is_hetero: Whether using heterogeneous graph
        loss_fn: Loss function (if None, uses F.cross_entropy)

    Returns:
        Average loss
    """
    model.train()
    optimizer.zero_grad()

    if is_hetero:
        data = data.to(device)
        out_dict = model(data.x_dict, data.edge_index_dict)

        tx_mask = data['transaction'].train_mask
        wallet_mask = data['wallet'].train_mask

        tx_train_labeled = tx_mask & (data['transaction'].y != 2)
        wallet_train_labeled = wallet_mask & (data['wallet'].y != 2)

        if tx_train_labeled.sum() > 0:
            if loss_fn is not None:
                tx_loss = loss_fn(
                    out_dict['transaction'][tx_train_labeled],
                    data['transaction'].y[tx_train_labeled]
                )
            else:
                tx_loss = F.cross_entropy(
                    out_dict['transaction'][tx_train_labeled],
                    data['transaction'].y[tx_train_labeled]
                )
        else:
            tx_loss = 0

        if wallet_train_labeled.sum() > 0:
            if loss_fn is not None:
                wallet_loss = loss_fn(
                    out_dict['wallet'][wallet_train_labeled],
                    data['wallet'].y[wallet_train_labeled]
                )
            else:
                wallet_loss = F.cross_entropy(
                    out_dict['wallet'][wallet_train_labeled],
                    data['wallet'].y[wallet_train_labeled]
                )
        else:
            wallet_loss = 0

        loss = tx_loss + wallet_loss

    else:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        train_labeled = data.train_mask & (data.y != 2)

        if loss_fn is not None:
            loss = loss_fn(out[train_labeled], data.y[train_labeled])
        else:
            loss = F.cross_entropy(out[train_labeled], data.y[train_labeled])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, device, is_hetero=False, node_type='transaction', mask_name='val_mask'):
    """
    Evaluate model

    Args:
        model: PyTorch model
        data: PyG Data or HeteroData object
        device: torch device
        is_hetero: Whether using heterogeneous graph
        node_type: Which node type to evaluate (for hetero graphs)
        mask_name: 'val_mask' or 'test_mask'

    Returns:
        Dictionary of metrics
    """
    model.eval()
    data = data.to(device)

    if is_hetero:
        out_dict = model(data.x_dict, data.edge_index_dict)
        out = out_dict[node_type]
        y_true = data[node_type].y
        mask = getattr(data[node_type], mask_name)
    else:
        out = model(data.x, data.edge_index)
        y_true = data.y
        mask = getattr(data, mask_name)

    # Only evaluate on labeled nodes
    mask_labeled = mask & (y_true != 2)

    if mask_labeled.sum() == 0:
        return {'accuracy': 0.0, 'f1': 0.0}

    pred = out[mask_labeled].argmax(dim=1)
    y_true_masked = y_true[mask_labeled]

    # Compute metrics
    acc = (pred == y_true_masked).sum().item() / mask_labeled.sum().item()

    # Convert to numpy for sklearn
    y_true_np = y_true_masked.cpu().numpy()
    pred_np = pred.cpu().numpy()

    # Handle case where we might only have one class in predictions
    try:
        f1_macro = f1_score(y_true_np, pred_np, average='macro', zero_division=0)
        f1_illicit = f1_score(y_true_np, pred_np, pos_label=0, average='binary', zero_division=0)
        f1_licit = f1_score(y_true_np, pred_np, pos_label=1, average='binary', zero_division=0)

        precision_illicit = precision_score(y_true_np, pred_np, pos_label=0, average='binary', zero_division=0)
        recall_illicit = recall_score(y_true_np, pred_np, pos_label=0, average='binary', zero_division=0)

        precision_licit = precision_score(y_true_np, pred_np, pos_label=1, average='binary', zero_division=0)
        recall_licit = recall_score(y_true_np, pred_np, pos_label=1, average='binary', zero_division=0)

    except Exception as e:
        print(f"Warning: Error computing metrics: {e}")
        f1_macro = 0.0
        f1_illicit = 0.0
        f1_licit = 0.0
        precision_illicit = 0.0
        recall_illicit = 0.0
        precision_licit = 0.0
        recall_licit = 0.0

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_illicit': f1_illicit,
        'f1_licit': f1_licit,
        'precision_illicit': precision_illicit,
        'recall_illicit': recall_illicit,
        'precision_licit': precision_licit,
        'recall_licit': recall_licit,
    }

    return metrics


def print_metrics(metrics: Dict, prefix: str = ""):
    """Pretty print metrics"""
    print(f"\n{prefix}Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"\n  Illicit class (0):")
    print(f"    F1: {metrics['f1_illicit']:.4f}")
    print(f"    Precision: {metrics['precision_illicit']:.4f}")
    print(f"    Recall: {metrics['recall_illicit']:.4f}")
    print(f"\n  Licit class (1):")
    print(f"    F1: {metrics['f1_licit']:.4f}")
    print(f"    Precision: {metrics['precision_licit']:.4f}")
    print(f"    Recall: {metrics['recall_licit']:.4f}")


def get_class_weights(y: torch.Tensor, num_classes: int = 2, alpha: float = 0.5) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets

    Args:
        y: Label tensor
        num_classes: Number of classes (excluding unknown)
        alpha: Scaling factor for weights (0=no weighting, 1=full inverse frequency)

    Returns:
        Class weights tensor
    """
    y_labeled = y[y != 2]

    if len(y_labeled) == 0:
        return torch.ones(num_classes)

    counts = torch.bincount(y_labeled, minlength=num_classes)
    weights = len(y_labeled) / (num_classes * counts.float())

    weights = 1.0 + alpha * (weights - 1.0)

    return weights


class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    Weighted cross entropy loss for imbalanced classes
    """

    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = weights

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weights)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance

    Lin et al. "Focal Loss for Dense Object Detection" (2017)
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for class imbalance (None or Tensor)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if should stop

        Args:
            value: Current metric value

        Returns:
            True if should stop training
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'max':
            improved = value > (self.best_value + self.min_delta)
        else:
            improved = value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


if __name__ == "__main__":
    print("Training utilities module")
    print("Use these functions in your training scripts")
