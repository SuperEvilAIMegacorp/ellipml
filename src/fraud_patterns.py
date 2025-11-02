"""
Fraud pattern detection for Bitcoin transaction graphs

Implements detection of three critical fraud patterns:
1. Peel chains - Money laundering via sequential splitting
2. Mixing services - Obfuscation hubs with symmetric high degree
3. Rapid dispersal - Hack/theft distribution patterns
"""

import torch
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict, Counter


class FraudPatternDetector:
    """Detects structural fraud patterns in Bitcoin transaction graphs"""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, timestamps: np.ndarray = None):
        """
        Args:
            edge_index: Graph edges [2, num_edges]
            num_nodes: Total number of nodes
            timestamps: Node timestamps for temporal analysis
        """
        self.edge_index = edge_index.cpu().numpy()
        self.num_nodes = num_nodes
        self.timestamps = timestamps

        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency lists for efficient pattern detection"""
        self.out_neighbors = defaultdict(list)
        self.in_neighbors = defaultdict(list)

        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            self.out_neighbors[src].append(dst)
            self.in_neighbors[dst].append(src)

        self.in_degree = np.array([len(self.in_neighbors[i]) for i in range(self.num_nodes)])
        self.out_degree = np.array([len(self.out_neighbors[i]) for i in range(self.num_nodes)])
        self.total_degree = self.in_degree + self.out_degree

    def detect_peel_chains(self) -> np.ndarray:
        """
        Detect peel chain patterns (money laundering)

        Pattern: Long sequences of 1-in-2-out transactions forming chains
        Score based on: chain participation, 1-in-2-out ratio, position in chain

        Returns:
            Array of peel chain scores [num_nodes]
        """
        scores = np.zeros(self.num_nodes)

        for node in range(self.num_nodes):
            in_deg = self.in_degree[node]
            out_deg = self.out_degree[node]

            if in_deg == 1 and out_deg == 2:
                chain_length = self._trace_chain_length(node)
                scores[node] = min(chain_length / 10.0, 1.0)

            elif in_deg == 1 and out_deg == 1:
                chain_length = self._trace_chain_length(node)
                scores[node] = min(chain_length / 15.0, 0.8)

        return scores

    def _trace_chain_length(self, start_node: int, max_depth: int = 50) -> int:
        """Trace chain length from starting node"""
        length = 1
        current = start_node

        for _ in range(max_depth):
            out_neighbors = self.out_neighbors[current]
            if len(out_neighbors) != 1 and len(out_neighbors) != 2:
                break

            if len(out_neighbors) == 1:
                current = out_neighbors[0]
            else:
                current = out_neighbors[0]

            if self.in_degree[current] != 1:
                break

            length += 1

        return length

    def detect_mixing_services(self) -> np.ndarray:
        """
        Detect mixing service hubs (obfuscation)

        Pattern: Very high symmetric degree (many ins + many outs)
        Score based on: min(in,out) degree, balance ratio, hub classification

        Returns:
            Array of mixing service scores [num_nodes]
        """
        scores = np.zeros(self.num_nodes)

        degree_threshold = np.percentile(self.total_degree[self.total_degree > 0], 95)

        for node in range(self.num_nodes):
            in_deg = self.in_degree[node]
            out_deg = self.out_degree[node]
            total_deg = self.total_degree[node]

            if total_deg < degree_threshold:
                continue

            min_deg = min(in_deg, out_deg)
            max_deg = max(in_deg, out_deg)

            if max_deg == 0:
                balance_ratio = 0
            else:
                balance_ratio = min_deg / max_deg

            hub_score = min(total_deg / 1000.0, 1.0)
            symmetry_score = balance_ratio

            scores[node] = hub_score * symmetry_score

        return scores

    def detect_rapid_dispersal(self) -> np.ndarray:
        """
        Detect rapid dispersal patterns (hack/theft distribution)

        Pattern: Sudden high out-degree after receiving, temporal clustering
        Score based on: out-degree spike, temporal concentration

        Returns:
            Array of rapid dispersal scores [num_nodes]
        """
        scores = np.zeros(self.num_nodes)

        for node in range(self.num_nodes):
            in_deg = self.in_degree[node]
            out_deg = self.out_degree[node]

            if in_deg == 0 or out_deg < 5:
                continue

            explosion_ratio = out_deg / max(in_deg, 1)

            if self.timestamps is not None:
                temporal_score = self._compute_temporal_burstiness(node)
            else:
                temporal_score = 0.5

            dispersal_score = min(explosion_ratio / 20.0, 1.0) * (0.5 + 0.5 * temporal_score)
            scores[node] = dispersal_score

        return scores

    def _compute_temporal_burstiness(self, node: int) -> float:
        """Compute temporal burstiness score for a node"""
        out_neighbors = self.out_neighbors[node]

        if len(out_neighbors) < 2:
            return 0.0

        node_time = self.timestamps[node]
        neighbor_times = [self.timestamps[n] for n in out_neighbors]

        time_diffs = np.abs(np.array(neighbor_times) - node_time)

        if len(time_diffs) < 2:
            return 0.0

        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)

        if mean_diff == 0:
            return 1.0

        burstiness = 1.0 - min(std_diff / (mean_diff + 1e-6), 1.0)

        return burstiness

    def compute_all_patterns(self) -> Dict[str, np.ndarray]:
        """
        Compute all fraud pattern scores

        Returns:
            Dictionary with pattern scores for each type
        """
        return {
            'peel_chain_score': self.detect_peel_chains(),
            'mixing_service_score': self.detect_mixing_services(),
            'rapid_dispersal_score': self.detect_rapid_dispersal()
        }


def augment_features_with_patterns(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    timestamps: np.ndarray = None
) -> torch.Tensor:
    """
    Augment node features with fraud pattern scores

    Args:
        x: Node features [num_nodes, num_features]
        edge_index: Graph edges [2, num_edges]
        timestamps: Optional timestamps for temporal analysis

    Returns:
        Augmented features [num_nodes, num_features + 3]
    """
    num_nodes = x.size(0)

    detector = FraudPatternDetector(edge_index, num_nodes, timestamps)
    patterns = detector.compute_all_patterns()

    pattern_features = np.stack([
        patterns['peel_chain_score'],
        patterns['mixing_service_score'],
        patterns['rapid_dispersal_score']
    ], axis=1)

    pattern_tensor = torch.tensor(pattern_features, dtype=torch.float)

    augmented_x = torch.cat([x, pattern_tensor], dim=1)

    return augmented_x


def analyze_pattern_distribution(
    patterns: Dict[str, np.ndarray],
    labels: np.ndarray,
    labeled_mask: np.ndarray
) -> Dict[str, Dict]:
    """
    Analyze pattern score distributions for illicit vs licit

    Args:
        patterns: Dictionary of pattern scores
        labels: Node labels (0=illicit, 1=licit, 2=unknown)
        labeled_mask: Mask for labeled nodes

    Returns:
        Dictionary with statistics for each pattern
    """
    results = {}

    for pattern_name, scores in patterns.items():
        illicit_mask = labeled_mask & (labels == 0)
        licit_mask = labeled_mask & (labels == 1)

        illicit_scores = scores[illicit_mask]
        licit_scores = scores[licit_mask]

        results[pattern_name] = {
            'illicit_mean': float(np.mean(illicit_scores)),
            'illicit_std': float(np.std(illicit_scores)),
            'illicit_median': float(np.median(illicit_scores)),
            'licit_mean': float(np.mean(licit_scores)),
            'licit_std': float(np.std(licit_scores)),
            'licit_median': float(np.median(licit_scores)),
            'separation': float(np.mean(illicit_scores) - np.mean(licit_scores))
        }

    return results
