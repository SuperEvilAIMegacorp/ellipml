"""
Phase 3: Visualization Strategy
Multi-level visualization system for graph exploration
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


class VisualizationEngine:
    """Multi-level visualization system for network analysis"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_scheme = {
            'illicit': '#d62728',  # red
            'licit': '#2ca02c',    # green
            'unknown': '#7f7f7f',   # gray
            'hub': '#ff7f0e',       # orange
            'leaf': '#1f77b4',      # blue
            'bridge': '#9467bd'     # purple
        }

    def plot_degree_distributions(self, degree_data: Dict, output_path: Optional[str] = None):
        """Create degree distribution plots for comparison"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Degree Distribution Analysis', fontsize=14)

        for idx, (graph_name, graph_data) in enumerate(degree_data.items()):
            row = idx // 3

            for col, degree_type in enumerate(['in_degree', 'out_degree', 'total_degree']):
                ax = axes[row, col]

                if degree_type in graph_data:
                    stats = graph_data[degree_type]

                    # Create histogram data
                    # Since we only have statistics, we'll create a box plot representation
                    positions = [1]
                    box_data = [[
                        stats.get('min', 0),
                        stats.get('p25', 0),
                        stats.get('median', 0),
                        stats.get('p75', 0),
                        stats.get('max', 0)
                    ]]

                    bp = ax.boxplot(box_data, positions=positions,
                                    widths=0.6, patch_artist=True,
                                    showmeans=True)

                    # Color the box
                    bp['boxes'][0].set_facecolor('#lightblue' if idx == 0 else 'lightgreen')

                    # Add statistics as text
                    ax.text(1.5, stats.get('p75', 0), f"Mean: {stats.get('mean', 0):.2f}")
                    ax.text(1.5, stats.get('median', 0), f"Std: {stats.get('std', 0):.2f}")
                    ax.text(1.5, stats.get('p25', 0), f"P95: {stats.get('p95', 0):.2f}")

                ax.set_title(f"{graph_name} - {degree_type.replace('_', ' ').title()}")
                ax.set_xlabel('Distribution')
                ax.set_ylabel('Degree')
                ax.set_xticks([])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_component_analysis(self, component_data: Dict, output_path: Optional[str] = None):
        """Visualize component structure of graphs"""

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Connected Component Analysis', fontsize=14)

        for idx, (graph_name, data) in enumerate(component_data.items()):
            row = idx

            # Weakly connected components
            ax = axes[row, 0]
            wcc_dist = data.get('component_size_distribution', {})

            metrics = [
                f"Num Components: {data.get('num_weakly_connected', 0)}",
                f"Largest: {data.get('largest_wcc_size', 0)}",
                f"Singletons: {wcc_dist.get('num_singleton', 0)}",
                f"Large (>100): {wcc_dist.get('num_large', 0)}"
            ]

            ax.text(0.1, 0.5, '\n'.join(metrics), transform=ax.transAxes,
                   fontsize=10, verticalalignment='center')
            ax.set_title(f"{graph_name} - Weakly Connected")
            ax.axis('off')

            # Strongly connected components
            ax = axes[row, 1]
            ax.text(0.1, 0.5,
                   f"Num Components: {data.get('num_strongly_connected', 0)}\n"
                   f"Largest: {data.get('largest_scc_size', 0)}",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='center')
            ax.set_title(f"{graph_name} - Strongly Connected")
            ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_motif_analysis(self, motif_data: Dict, output_path: Optional[str] = None):
        """Visualize motif patterns found in graphs"""

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle('Structural Motif Analysis', fontsize=14)

        # Transaction motifs
        ax = axes[0]
        tx_motifs = motif_data.get('transaction_motifs', {})

        motif_names = ['Linear Chains', 'Forks', 'Merges', 'Cycles']
        motif_counts = [
            tx_motifs.get('linear_chain_count', 0),
            tx_motifs.get('fork_count', 0),
            tx_motifs.get('merge_count', 0),
            tx_motifs.get('cycle_count', 0)
        ]

        bars = ax.bar(motif_names, motif_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_title('Transaction Graph Motifs')
        ax.set_ylabel('Count')
        ax.set_xlabel('Motif Type')

        # Add value labels on bars
        for bar, count in zip(bars, motif_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom')

        # Bipartite motifs
        ax = axes[1]
        bi_motifs = motif_data.get('bipartite_motifs', {})

        motif_names = ['Fan-in', 'Fan-out', 'Pass-through', 'Mixing']
        motif_counts = [
            bi_motifs.get('fan_in_count', 0),
            bi_motifs.get('fan_out_count', 0),
            bi_motifs.get('pass_through_count', 0),
            bi_motifs.get('mixing_count', 0)
        ]

        bars = ax.bar(motif_names, motif_counts, color=['#9467bd', '#8c564b', '#e377c2', '#bcbd22'])
        ax.set_title('Bipartite Graph Motifs')
        ax.set_ylabel('Count')
        ax.set_xlabel('Motif Type')

        for bar, count in zip(bars, motif_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_role_distribution(self, role_data: Dict, output_path: Optional[str] = None):
        """Visualize distribution of structural roles"""

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle('Structural Role Distribution', fontsize=14)

        for idx, (graph_name, role_counts) in enumerate(role_data.items()):
            ax = axes[idx]

            if role_counts:
                roles = list(role_counts.keys())
                counts = list(role_counts.values())

                # Create pie chart
                colors = [self.color_scheme.get(role, '#gray') for role in roles]
                wedges, texts, autotexts = ax.pie(counts, labels=roles, colors=colors,
                                                   autopct='%1.1f%%', startangle=90)

                ax.set_title(f"{graph_name.replace('_', ' ').title()}")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_ego_network(self, graph: nx.Graph, center_node: Any,
                        max_hops: int = 2, labels: Optional[Dict] = None,
                        tags: Optional[Dict] = None, output_path: Optional[str] = None):
        """Visualize ego network around a specific node"""

        # Extract ego network
        ego_nodes = {center_node}
        current_layer = {center_node}

        for hop in range(max_hops):
            next_layer = set()
            for node in current_layer:
                next_layer.update(graph.neighbors(node))
            ego_nodes.update(next_layer)
            current_layer = next_layer

        ego_graph = graph.subgraph(ego_nodes)

        # Setup plot
        plt.figure(figsize=self.figsize)

        # Determine layout
        pos = nx.spring_layout(ego_graph, k=1/np.sqrt(len(ego_nodes)), iterations=50)

        # Determine node colors
        node_colors = []
        for node in ego_graph.nodes():
            if labels and node in labels:
                if labels[node] in ['1', 1]:
                    node_colors.append(self.color_scheme['illicit'])
                elif labels[node] in ['2', 2]:
                    node_colors.append(self.color_scheme['licit'])
                else:
                    node_colors.append(self.color_scheme['unknown'])
            else:
                node_colors.append(self.color_scheme['unknown'])

        # Determine node sizes
        node_sizes = []
        for node in ego_graph.nodes():
            if node == center_node:
                node_sizes.append(500)
            else:
                node_sizes.append(100)

        # Draw network
        nx.draw_networkx_nodes(ego_graph, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(ego_graph, pos, alpha=0.3, edge_color='gray')

        # Add center node label
        nx.draw_networkx_labels(ego_graph, pos, {center_node: str(center_node)[:8]},
                               font_size=8)

        plt.title(f"Ego Network (center: {str(center_node)[:20]}, hops: {max_hops})")
        plt.axis('off')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_tag_distribution(self, tags: Dict, tag_type: str = 'structural',
                             output_path: Optional[str] = None):
        """Plot distribution of tags across nodes"""

        # Collect all unique tags
        tag_counts = {}
        for node, node_tags in tags.get(tag_type, {}).items():
            for tag in node_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if not tag_counts:
            print(f"No tags found for type: {tag_type}")
            return

        # Sort by frequency
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        plt.figure(figsize=self.figsize)

        tags = [t[0] for t in sorted_tags]
        counts = [t[1] for t in sorted_tags]

        bars = plt.bar(range(len(tags)), counts, color='steelblue')
        plt.xticks(range(len(tags)), tags, rotation=45, ha='right')
        plt.xlabel('Tag')
        plt.ylabel('Count')
        plt.title(f'{tag_type.title()} Tag Distribution (Top 20)')

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_temporal_heatmap(self, wallet_features: pd.DataFrame,
                             sample_size: int = 100,
                             output_path: Optional[str] = None):
        """Create temporal activity heatmap"""

        # Sample wallets
        wallet_sample = wallet_features.drop_duplicates(subset='address').sample(
            n=min(sample_size, len(wallet_features)), random_state=42
        )

        # Create activity matrix
        activity_matrix = np.zeros((len(wallet_sample), 49))

        for idx, (_, wallet) in enumerate(wallet_sample.iterrows()):
            # Get all timesteps this wallet appears in
            wallet_all = wallet_features[wallet_features['address'] == wallet['address']]
            for _, appearance in wallet_all.iterrows():
                timestep = int(appearance['Time step']) - 1  # 0-indexed
                activity_matrix[idx, timestep] = 1

        # Plot heatmap
        plt.figure(figsize=(15, 8))

        sns.heatmap(activity_matrix, cmap='YlOrRd', cbar_kws={'label': 'Active'},
                   xticklabels=range(1, 50), yticklabels=False)

        plt.xlabel('Time Step')
        plt.ylabel(f'Wallets (sample of {sample_size})')
        plt.title('Temporal Activity Heatmap')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()

    def create_summary_dashboard(self, stats: Dict, output_path: Optional[str] = None):
        """Create a summary dashboard with key metrics"""

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Overall statistics
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')

        summary_text = f"""
        Network Overview:
        • Transaction nodes: {stats.get('tx_nodes', 0):,}
        • Transaction edges: {stats.get('tx_edges', 0):,}
        • Wallet nodes: {stats.get('wallet_nodes', 0):,}
        • Wallet edges: {stats.get('wallet_edges', 0):,}
        • Bipartite edges: {stats.get('bipartite_edges', 0):,}
        """

        ax1.text(0.1, 0.5, summary_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='center')

        # Component statistics
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        comp_text = f"""
        Component Structure:
        • TX WCC: {stats.get('tx_wcc', 0):,}
        • TX SCC: {stats.get('tx_scc', 0):,}
        • Wallet WCC: {stats.get('wallet_wcc', 0):,}
        • Wallet SCC: {stats.get('wallet_scc', 0):,}
        """
        ax2.text(0.1, 0.5, comp_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='center')

        # Motif statistics
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        motif_text = f"""
        Motif Patterns:
        • Linear chains: {stats.get('linear_chains', 0):,}
        • Forks: {stats.get('forks', 0):,}
        • Merges: {stats.get('merges', 0):,}
        • Pass-throughs: {stats.get('pass_through', 0):,}
        """
        ax3.text(0.1, 0.5, motif_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='center')

        # Role distribution
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        role_text = f"""
        Role Distribution:
        • Hubs: {stats.get('hubs', 0):,}
        • Leaves: {stats.get('leaves', 0):,}
        • Bridges: {stats.get('bridges', 0):,}
        • Intermediate: {stats.get('intermediate', 0):,}
        """
        ax4.text(0.1, 0.5, role_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center')

        # Tag summary
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        tag_text = f"""
        Tag Generation:
        • Structural tags generated: {stats.get('structural_tags', 0):,}
        • Behavioral tags generated: {stats.get('behavioral_tags', 0):,}
        • Propagated tags generated: {stats.get('propagated_tags', 0):,}
        """
        ax5.text(0.1, 0.5, tag_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='center')

        plt.suptitle('Bitcoin Network Structure Analysis Dashboard', fontsize=16)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()