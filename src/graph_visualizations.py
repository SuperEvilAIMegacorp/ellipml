"""
Graph visualization functions for fraud detection analysis

Generates publication-quality visualizations for error analysis,
pattern validation, and model explainability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple
import torch


class FraudGraphVisualizer:
    """Generate visualizations for fraud detection graphs"""

    def __init__(self, data, predictions=None, pattern_scores=None):
        """
        Args:
            data: PyG Data object with node features, edges, labels
            predictions: Model predictions [num_nodes]
            pattern_scores: Dictionary of pattern scores (optional)
        """
        self.data = data
        self.predictions = predictions
        self.pattern_scores = pattern_scores

        self.labels = data.y.numpy()
        self.edge_index = data.edge_index.numpy()
        self.test_mask = data.test_mask.numpy()

    def plot_confusion_subgraphs(self, save_dir: str, max_nodes_per_type: int = 30):
        """
        Visualize error analysis: TP, FP, FN, TN subgraphs

        Args:
            save_dir: Directory to save plots
            max_nodes_per_type: Maximum nodes to visualize per category
        """
        if self.predictions is None:
            print("No predictions provided, skipping confusion analysis")
            return

        test_indices = np.where(self.test_mask)[0]
        test_labels = self.labels[test_indices]
        test_preds = self.predictions[test_indices]

        tp_idx = test_indices[(test_preds == 0) & (test_labels == 0)]
        fp_idx = test_indices[(test_preds == 0) & (test_labels == 1)]
        fn_idx = test_indices[(test_preds == 1) & (test_labels == 0)]
        tn_idx = test_indices[(test_preds == 1) & (test_labels == 1)]

        categories = [
            ("True Positives", tp_idx, "Correctly identified illicit"),
            ("False Positives", fp_idx, "Licit flagged as illicit"),
            ("False Negatives", fn_idx, "Missed illicit transactions"),
            ("True Negatives", tn_idx[:max_nodes_per_type], "Correctly identified licit")
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        axes = axes.flatten()

        for idx, (title, nodes, subtitle) in enumerate(categories):
            if len(nodes) == 0:
                axes[idx].text(0.5, 0.5, f"No {title}", ha='center', va='center')
                axes[idx].set_title(f"{title}\n{subtitle}")
                axes[idx].axis('off')
                continue

            sample_nodes = nodes[:max_nodes_per_type]
            subgraph = self._extract_subgraph(sample_nodes, k_hop=1)

            self._plot_subgraph(
                axes[idx],
                subgraph,
                title=f"{title} (n={len(nodes)})\n{subtitle}",
                highlight_nodes=sample_nodes
            )

        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_subgraphs.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion analysis: TP={len(tp_idx)}, FP={len(fp_idx)}, "
              f"FN={len(fn_idx)}, TN={len(tn_idx)}")

    def plot_peel_chain_subgraphs(self, save_dir: str, num_chains: int = 6):
        """
        Visualize high-scoring peel chain patterns

        Args:
            save_dir: Directory to save plots
            num_chains: Number of peel chains to visualize
        """
        if self.pattern_scores is None or 'peel_chain_score' not in self.pattern_scores:
            print("No peel chain scores provided, skipping")
            return

        peel_scores = self.pattern_scores['peel_chain_score']
        high_scorers = np.where(peel_scores > 0.5)[0]

        if len(high_scorers) == 0:
            print("No high-scoring peel chains found")
            return

        sorted_idx = high_scorers[np.argsort(-peel_scores[high_scorers])]
        top_chains = sorted_idx[:num_chains]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, node in enumerate(top_chains):
            if idx >= len(axes):
                break

            chain_nodes = self._trace_chain(node)
            subgraph = self._extract_subgraph(np.array(chain_nodes), k_hop=0)

            illicit_count = sum(self.labels[n] == 0 for n in chain_nodes if self.labels[n] != 2)
            total_labeled = sum(self.labels[n] != 2 for n in chain_nodes)

            title = f"Peel Chain (score={peel_scores[node]:.2f})\n"
            title += f"Illicit: {illicit_count}/{total_labeled} nodes"

            nodes_in_graph = list(subgraph.nodes())
            highlight = [node] if node in nodes_in_graph else []

            self._plot_subgraph(
                axes[idx],
                subgraph,
                title=title,
                highlight_nodes=highlight,
                layout='hierarchical'
            )

        for idx in range(len(top_chains), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/peel_chain_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualized {len(top_chains)} peel chains")

    def plot_degree_distribution(self, save_dir: str):
        """
        Plot degree distribution with fraud rate overlay

        Args:
            save_dir: Directory to save plot
        """
        in_degree = np.zeros(len(self.labels))
        out_degree = np.zeros(len(self.labels))

        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            out_degree[src] += 1
            in_degree[dst] += 1

        total_degree = in_degree + out_degree

        labeled_mask = self.labels != 2
        illicit_mask = self.labels == 0

        degree_bins = np.array([0, 1, 2, 3, 5, 10, 20, 50, 100, 500, 10000])
        bin_counts = []
        illicit_rates = []

        for i in range(len(degree_bins) - 1):
            low, high = degree_bins[i], degree_bins[i + 1]
            in_bin = (total_degree >= low) & (total_degree < high) & labeled_mask

            if in_bin.sum() > 0:
                bin_counts.append(in_bin.sum())
                illicit_rates.append((illicit_mask & in_bin).sum() / in_bin.sum() * 100)
            else:
                bin_counts.append(0)
                illicit_rates.append(0)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        bin_labels = [f"{degree_bins[i]}-{degree_bins[i+1]-1}" for i in range(len(degree_bins)-1)]
        x_pos = np.arange(len(bin_labels))

        color1 = 'steelblue'
        ax1.bar(x_pos, bin_counts, alpha=0.7, color=color1, label='Node count')
        ax1.set_xlabel('Total Degree Range', fontsize=12)
        ax1.set_ylabel('Number of Nodes', color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_yscale('log')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(bin_labels, rotation=45, ha='right')

        ax2 = ax1.twinx()
        color2 = 'crimson'
        ax2.plot(x_pos, illicit_rates, color=color2, marker='o', linewidth=2, markersize=8, label='Illicit rate')
        ax2.set_ylabel('Illicit Rate (%)', color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axhline(y=9.76, color='gray', linestyle='--', alpha=0.5, label='Baseline (9.76%)')

        ax1.set_title('Degree Distribution and Fraud Rate', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/degree_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Degree distribution: max_degree={total_degree.max():.0f}")

    def plot_prediction_heatmap(self, save_dir: str, sample_size: int = 500, confidence_threshold: float = 0.7):
        """
        Plot prediction confidence heatmap on graph sample

        Args:
            save_dir: Directory to save plot
            sample_size: Number of nodes to sample
            confidence_threshold: Threshold for high confidence
        """
        if self.predictions is None:
            print("No predictions provided, skipping heatmap")
            return

        test_indices = np.where(self.test_mask)[0]
        if len(test_indices) > sample_size:
            sample_indices = np.random.choice(test_indices, sample_size, replace=False)
        else:
            sample_indices = test_indices

        subgraph = self._extract_subgraph(sample_indices, k_hop=1)

        fig, ax = plt.subplots(figsize=(14, 14))

        self._plot_subgraph(
            ax,
            subgraph,
            title=f"Model Prediction Confidence (n={len(sample_indices)})",
            use_predictions=True
        )

        plt.tight_layout()
        plt.savefig(f"{save_dir}/prediction_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Prediction heatmap: sampled {len(sample_indices)} nodes")

    def plot_heterogeneous_structure(self, save_dir: str, hetero_data=None, sample_txs: int = 200):
        """
        Visualize transaction-wallet heterogeneous graph structure

        Args:
            save_dir: Directory to save plot
            hetero_data: HeteroData object
            sample_txs: Number of transactions to sample
        """
        if hetero_data is None:
            print("No heterogeneous data provided, skipping")
            return

        tx_test_mask = hetero_data['transaction'].test_mask.cpu().numpy()
        test_tx_indices = np.where(tx_test_mask)[0]

        if len(test_tx_indices) > sample_txs:
            sample_tx = np.random.choice(test_tx_indices, sample_txs, replace=False)
        else:
            sample_tx = test_tx_indices

        edge_types = list(hetero_data.edge_types)
        tx_wallet_edge_type = None
        for et in edge_types:
            if et[0] == 'transaction' and et[2] == 'wallet':
                tx_wallet_edge_type = et
                break

        if tx_wallet_edge_type is None:
            print("No transaction-wallet edges found")
            return

        tx_wallet_edges = hetero_data[tx_wallet_edge_type].edge_index.cpu().numpy()

        connected_wallets = set()
        for tx_idx in sample_tx:
            wallet_neighbors = tx_wallet_edges[1, tx_wallet_edges[0] == tx_idx]
            connected_wallets.update(wallet_neighbors[:5])

        G = nx.Graph()

        for tx_idx in sample_tx:
            G.add_node(f"tx_{tx_idx}", node_type='transaction', idx=tx_idx)

        for wallet_idx in list(connected_wallets)[:100]:
            G.add_node(f"wallet_{wallet_idx}", node_type='wallet', idx=wallet_idx)

        for i in range(tx_wallet_edges.shape[1]):
            tx_idx, wallet_idx = tx_wallet_edges[0, i], tx_wallet_edges[1, i]
            if tx_idx in sample_tx and wallet_idx in connected_wallets:
                G.add_edge(f"tx_{tx_idx}", f"wallet_{wallet_idx}")

        fig, ax = plt.subplots(figsize=(16, 12))

        pos = nx.bipartite_layout(
            G,
            [n for n, d in G.nodes(data=True) if d.get('node_type') == 'transaction']
        )

        tx_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'transaction']
        wallet_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'wallet']

        tx_labels = hetero_data['transaction'].y.cpu().numpy()
        tx_colors = []
        for n in tx_nodes:
            idx = G.nodes[n]['idx']
            if tx_labels[idx] == 0:
                tx_colors.append('#e74c3c')
            elif tx_labels[idx] == 1:
                tx_colors.append('#2ecc71')
            else:
                tx_colors.append('#95a5a6')

        nx.draw_networkx_nodes(G, pos, nodelist=tx_nodes, node_color=tx_colors,
                              node_size=100, alpha=0.8, ax=ax, label='Transactions')
        nx.draw_networkx_nodes(G, pos, nodelist=wallet_nodes, node_color='#3498db',
                              node_size=80, alpha=0.6, node_shape='s', ax=ax, label='Wallets')
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)

        ax.set_title(f"Heterogeneous Graph: Transactions-Wallets\n"
                    f"{len(tx_nodes)} transactions, {len(wallet_nodes)} wallets",
                    fontsize=14, fontweight='bold')

        red_patch = mpatches.Patch(color='#e74c3c', label='Illicit')
        green_patch = mpatches.Patch(color='#2ecc71', label='Licit')
        gray_patch = mpatches.Patch(color='#95a5a6', label='Unknown')
        blue_patch = mpatches.Patch(color='#3498db', label='Wallets')

        ax.legend(handles=[red_patch, green_patch, gray_patch, blue_patch], loc='upper right')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/heterogeneous_structure.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Heterogeneous graph: {len(tx_nodes)} transactions, {len(wallet_nodes)} wallets")

    def _extract_subgraph(self, seed_nodes: np.ndarray, k_hop: int = 1) -> nx.DiGraph:
        """Extract k-hop neighborhood subgraph around seed nodes"""
        G = nx.DiGraph()

        nodes_to_include = set(seed_nodes)

        if k_hop > 0:
            for _ in range(k_hop):
                new_nodes = set()
                for i in range(self.edge_index.shape[1]):
                    src, dst = self.edge_index[0, i], self.edge_index[1, i]
                    if src in nodes_to_include:
                        new_nodes.add(dst)
                    if dst in nodes_to_include:
                        new_nodes.add(src)
                nodes_to_include.update(new_nodes)

        for node in nodes_to_include:
            G.add_node(node, label=self.labels[node])

        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            if src in nodes_to_include and dst in nodes_to_include:
                G.add_edge(src, dst)

        return G

    def _plot_subgraph(self, ax, G: nx.DiGraph, title: str, highlight_nodes=None,
                      layout='spring', use_predictions=False):
        """Plot a subgraph with appropriate styling"""
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, "Empty subgraph", ha='center', va='center')
            ax.set_title(title)
            ax.axis('off')
            return

        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif layout == 'hierarchical':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)

        node_colors = []
        for node in G.nodes():
            label = self.labels[node]

            if use_predictions and self.predictions is not None:
                pred = self.predictions[node]
                if pred == 0:
                    node_colors.append('#e74c3c')
                else:
                    node_colors.append('#2ecc71')
            else:
                if label == 0:
                    node_colors.append('#e74c3c')
                elif label == 1:
                    node_colors.append('#2ecc71')
                else:
                    node_colors.append('#95a5a6')

        node_sizes = [100 if highlight_nodes is None or n not in highlight_nodes else 300
                     for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                              alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.0, arrows=True,
                              arrowsize=10, ax=ax)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        red_patch = mpatches.Patch(color='#e74c3c', label='Illicit')
        green_patch = mpatches.Patch(color='#2ecc71', label='Licit')
        gray_patch = mpatches.Patch(color='#95a5a6', label='Unknown')

        ax.legend(handles=[red_patch, green_patch, gray_patch], loc='upper right', fontsize=8)

    def _trace_chain(self, start_node: int, max_depth: int = 50) -> List[int]:
        """Trace peel chain from starting node"""
        chain = [start_node]
        current = start_node

        for _ in range(max_depth):
            out_neighbors = self.edge_index[1, self.edge_index[0] == current]

            if len(out_neighbors) == 0:
                break

            next_node = out_neighbors[0]
            chain.append(next_node)

            in_neighbors = self.edge_index[0, self.edge_index[1] == next_node]
            if len(in_neighbors) != 1:
                break

            current = next_node

        return chain
