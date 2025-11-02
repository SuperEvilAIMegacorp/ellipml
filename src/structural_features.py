import torch
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Tuple
import json


class StructuralFeatureEngineer:
    """
    Integrates structural analysis features with raw node features
    """

    def __init__(self, analysis_path: str = None):
        self.analysis_path = analysis_path
        self.structural_stats = None

        if analysis_path and Path(analysis_path).exists():
            with open(analysis_path, 'r') as f:
                self.structural_stats = json.load(f)

    def compute_degree_features(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute degree-based features from edge index

        Returns:
            Tensor of shape [num_nodes, 6] with degree features
        """
        edge_index_np = edge_index.cpu().numpy()

        in_degree = np.bincount(edge_index_np[1], minlength=num_nodes)
        out_degree = np.bincount(edge_index_np[0], minlength=num_nodes)
        total_degree = in_degree + out_degree

        degree_ratio = np.where(total_degree > 0,
                               in_degree / total_degree,
                               0)

        log_in_degree = np.log1p(in_degree)
        log_out_degree = np.log1p(out_degree)

        features = np.stack([
            in_degree,
            out_degree,
            total_degree,
            degree_ratio,
            log_in_degree,
            log_out_degree
        ], axis=1).astype(np.float32)

        return torch.from_numpy(features)

    def compute_centrality_features(self, edge_index: torch.Tensor,
                                   num_nodes: int,
                                   sample_size: int = 5000) -> torch.Tensor:
        """
        Approximate centrality measures using sampling

        Returns:
            Tensor of shape [num_nodes, 3]
        """
        edge_list = edge_index.t().cpu().numpy()

        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)

        if num_nodes > sample_size:
            sampled_nodes = np.random.choice(num_nodes, sample_size, replace=False)
            G_sample = G.subgraph(sampled_nodes)
        else:
            G_sample = G
            sampled_nodes = list(range(num_nodes))

        try:
            pagerank = nx.pagerank(G_sample, max_iter=50)
        except:
            pagerank = {n: 1.0/len(G_sample) for n in G_sample.nodes()}

        try:
            clustering = nx.clustering(G_sample.to_undirected())
        except:
            clustering = {n: 0.0 for n in G_sample.nodes()}

        core_numbers = nx.core_number(G_sample.to_undirected())

        pagerank_full = np.zeros(num_nodes, dtype=np.float32)
        clustering_full = np.zeros(num_nodes, dtype=np.float32)
        core_full = np.zeros(num_nodes, dtype=np.float32)

        for i, node in enumerate(sampled_nodes):
            pagerank_full[node] = pagerank.get(node, 0.0)
            clustering_full[node] = clustering.get(node, 0.0)
            core_full[node] = core_numbers.get(node, 0)

        features = np.stack([
            pagerank_full,
            clustering_full,
            core_full
        ], axis=1)

        return torch.from_numpy(features)

    def compute_component_features(self, edge_index: torch.Tensor,
                                   num_nodes: int) -> torch.Tensor:
        """
        Compute connected component features

        Returns:
            Tensor of shape [num_nodes, 3]
        """
        edge_list = edge_index.t().cpu().numpy()

        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)

        weak_components = list(nx.weakly_connected_components(G))
        strong_components = list(nx.strongly_connected_components(G))

        weak_comp_id = np.zeros(num_nodes, dtype=np.float32)
        weak_comp_size = np.zeros(num_nodes, dtype=np.float32)
        strong_comp_size = np.zeros(num_nodes, dtype=np.float32)

        for idx, comp in enumerate(weak_components):
            size = len(comp)
            for node in comp:
                weak_comp_id[node] = idx
                weak_comp_size[node] = size

        for comp in strong_components:
            size = len(comp)
            for node in comp:
                strong_comp_size[node] = size

        weak_comp_size_log = np.log1p(weak_comp_size)

        features = np.stack([
            weak_comp_id / (len(weak_components) + 1),
            weak_comp_size_log,
            strong_comp_size
        ], axis=1)

        return torch.from_numpy(features)

    def augment_features(self, x: torch.Tensor,
                        edge_index: torch.Tensor,
                        compute_expensive: bool = False) -> torch.Tensor:
        """
        Augment existing features with structural features

        Args:
            x: Original feature tensor [num_nodes, feat_dim]
            edge_index: Edge index tensor [2, num_edges]
            compute_expensive: Whether to compute expensive features like centrality

        Returns:
            Augmented feature tensor
        """
        num_nodes = x.size(0)

        degree_feats = self.compute_degree_features(edge_index, num_nodes)

        features_to_concat = [x, degree_feats]

        if compute_expensive:
            component_feats = self.compute_component_features(edge_index, num_nodes)
            centrality_feats = self.compute_centrality_features(edge_index, num_nodes)

            features_to_concat.extend([component_feats, centrality_feats])

        augmented = torch.cat(features_to_concat, dim=1)

        return augmented

    def get_role_labels(self, edge_index: torch.Tensor,
                       num_nodes: int,
                       p95_threshold: int = None) -> torch.Tensor:
        """
        Generate role labels: hub, intermediate, leaf

        Returns:
            Tensor of shape [num_nodes] with role labels
        """
        edge_index_np = edge_index.cpu().numpy()
        total_degree = np.bincount(edge_index_np[0], minlength=num_nodes) + \
                       np.bincount(edge_index_np[1], minlength=num_nodes)

        if p95_threshold is None:
            p95_threshold = np.percentile(total_degree[total_degree > 0], 95)

        roles = np.zeros(num_nodes, dtype=np.long)

        roles[total_degree == 1] = 2
        roles[(total_degree > 1) & (total_degree <= p95_threshold)] = 1
        roles[total_degree > p95_threshold] = 0

        return torch.from_numpy(roles)
