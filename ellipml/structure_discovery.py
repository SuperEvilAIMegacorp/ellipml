"""
Phase 1: Structure Discovery
Graph decomposition, motif mining, and role detection
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json
from pathlib import Path


class StructuralAnalyzer:
    """Analyze graph structure and extract topological features"""

    def __init__(self, data_dir: str = ".."):
        self.data_dir = Path(data_dir)
        self.tx_graph = None
        self.wallet_graph = None
        self.bipartite_graph = None
        self.components = {}
        self.structural_features = {}

    def load_graphs(self):
        """Load and construct graph objects from raw data"""

        # Load transaction graph
        tx_edges = pd.read_csv(self.data_dir / "ellipticpp/txs_edgelist.csv")
        self.tx_graph = nx.from_pandas_edgelist(
            tx_edges,
            source='txId1',
            target='txId2',
            create_using=nx.DiGraph()
        )

        # Load wallet graph (sample for memory efficiency)
        wallet_edges = pd.read_csv(self.data_dir / "ellipticpp/AddrAddr_edgelist.csv")
        # Sample 500K edges for tractability
        if len(wallet_edges) > 500000:
            wallet_edges = wallet_edges.sample(n=500000, random_state=42)

        self.wallet_graph = nx.from_pandas_edgelist(
            wallet_edges,
            source='input_address',
            target='output_address',
            create_using=nx.DiGraph()
        )

        # Create bipartite graph
        self.bipartite_graph = nx.Graph()

        # Add wallet->tx edges
        addr_tx = pd.read_csv(self.data_dir / "ellipticpp/AddrTx_edgelist.csv")
        for _, row in addr_tx.iterrows():
            self.bipartite_graph.add_edge(
                ('wallet', row['input_address']),
                ('tx', row['txId']),
                relation='input'
            )

        # Add tx->wallet edges
        tx_addr = pd.read_csv(self.data_dir / "ellipticpp/TxAddr_edgelist.csv")
        for _, row in tx_addr.iterrows():
            self.bipartite_graph.add_edge(
                ('tx', row['txId']),
                ('wallet', row['output_address']),
                relation='output'
            )

        return {
            'tx_nodes': self.tx_graph.number_of_nodes(),
            'tx_edges': self.tx_graph.number_of_edges(),
            'wallet_nodes': self.wallet_graph.number_of_nodes(),
            'wallet_edges': self.wallet_graph.number_of_edges(),
            'bipartite_nodes': self.bipartite_graph.number_of_nodes(),
            'bipartite_edges': self.bipartite_graph.number_of_edges()
        }

    def analyze_components(self):
        """Analyze connected components in each graph"""

        results = {}

        # Transaction graph components
        tx_wcc = list(nx.weakly_connected_components(self.tx_graph))
        tx_scc = list(nx.strongly_connected_components(self.tx_graph))

        results['tx_graph'] = {
            'num_weakly_connected': len(tx_wcc),
            'largest_wcc_size': len(max(tx_wcc, key=len)),
            'num_strongly_connected': len(tx_scc),
            'largest_scc_size': len(max(tx_scc, key=len)),
            'component_size_distribution': self._get_size_distribution(tx_wcc)
        }

        # Wallet graph components
        wallet_wcc = list(nx.weakly_connected_components(self.wallet_graph))
        wallet_scc = list(nx.strongly_connected_components(self.wallet_graph))

        results['wallet_graph'] = {
            'num_weakly_connected': len(wallet_wcc),
            'largest_wcc_size': len(max(wallet_wcc, key=len)),
            'num_strongly_connected': len(wallet_scc),
            'largest_scc_size': len(max(wallet_scc, key=len)),
            'component_size_distribution': self._get_size_distribution(wallet_wcc)
        }

        self.components = results
        return results

    def compute_degree_distributions(self):
        """Compute degree distributions for analysis"""

        results = {}

        # Transaction graph
        tx_in_degrees = dict(self.tx_graph.in_degree())
        tx_out_degrees = dict(self.tx_graph.out_degree())
        tx_total_degrees = dict(self.tx_graph.degree())

        results['tx_graph'] = {
            'in_degree': self._compute_distribution_stats(list(tx_in_degrees.values())),
            'out_degree': self._compute_distribution_stats(list(tx_out_degrees.values())),
            'total_degree': self._compute_distribution_stats(list(tx_total_degrees.values()))
        }

        # Wallet graph
        wallet_in_degrees = dict(self.wallet_graph.in_degree())
        wallet_out_degrees = dict(self.wallet_graph.out_degree())
        wallet_total_degrees = dict(self.wallet_graph.degree())

        results['wallet_graph'] = {
            'in_degree': self._compute_distribution_stats(list(wallet_in_degrees.values())),
            'out_degree': self._compute_distribution_stats(list(wallet_out_degrees.values())),
            'total_degree': self._compute_distribution_stats(list(wallet_total_degrees.values()))
        }

        return results

    def extract_motifs(self, sample_size: int = 1000):
        """Extract and count common motifs in the graph"""

        # Sample nodes for motif extraction (full graph too large)
        tx_nodes = list(self.tx_graph.nodes())[:sample_size]

        motifs = {
            'linear_chains': [],
            'forks': [],
            'merges': [],
            'cycles': []
        }

        for node in tx_nodes:
            # Linear chain detection (1-in, 1-out pattern)
            in_deg = self.tx_graph.in_degree(node)
            out_deg = self.tx_graph.out_degree(node)

            if in_deg == 1 and out_deg == 1:
                motifs['linear_chains'].append(node)

            # Fork detection (1-in, many-out)
            elif in_deg == 1 and out_deg > 2:
                motifs['forks'].append({
                    'node': node,
                    'fan_out': out_deg
                })

            # Merge detection (many-in, 1-out)
            elif in_deg > 2 and out_deg == 1:
                motifs['merges'].append({
                    'node': node,
                    'fan_in': in_deg
                })

            # Simple cycle detection (3-cycles)
            for neighbor in self.tx_graph.neighbors(node):
                for neighbor2 in self.tx_graph.neighbors(neighbor):
                    if node in self.tx_graph.neighbors(neighbor2):
                        cycle = tuple(sorted([node, neighbor, neighbor2]))
                        if cycle not in motifs['cycles']:
                            motifs['cycles'].append(cycle)

        # Bipartite motifs
        bipartite_motifs = self._extract_bipartite_motifs(sample_size=500)

        return {
            'transaction_motifs': {
                'linear_chain_count': len(motifs['linear_chains']),
                'fork_count': len(motifs['forks']),
                'merge_count': len(motifs['merges']),
                'cycle_count': len(motifs['cycles']),
                'avg_fork_fanout': np.mean([f['fan_out'] for f in motifs['forks']]) if motifs['forks'] else 0,
                'avg_merge_fanin': np.mean([m['fan_in'] for m in motifs['merges']]) if motifs['merges'] else 0
            },
            'bipartite_motifs': bipartite_motifs
        }

    def detect_structural_roles(self):
        """Identify structural roles of nodes"""

        roles = {
            'tx_graph': {},
            'wallet_graph': {}
        }

        # Transaction graph roles
        tx_degrees = dict(self.tx_graph.degree())
        tx_degree_values = list(tx_degrees.values())
        tx_p95 = np.percentile(tx_degree_values, 95)

        # Compute centrality for sample (full graph too expensive)
        tx_sample = list(self.tx_graph.nodes())[:5000]
        tx_subgraph = self.tx_graph.subgraph(tx_sample)
        tx_betweenness = nx.betweenness_centrality(tx_subgraph, k=100)

        for node in tx_sample:
            if tx_degrees[node] >= tx_p95:
                roles['tx_graph'][node] = 'hub'
            elif tx_degrees[node] == 1:
                roles['tx_graph'][node] = 'leaf'
            elif tx_betweenness.get(node, 0) > 0.01:
                roles['tx_graph'][node] = 'bridge'
            else:
                roles['tx_graph'][node] = 'intermediate'

        # Wallet graph roles
        wallet_degrees = dict(self.wallet_graph.degree())
        wallet_degree_values = list(wallet_degrees.values())
        wallet_p95 = np.percentile(wallet_degree_values, 95)

        # Sample for centrality
        wallet_sample = list(self.wallet_graph.nodes())[:5000]
        wallet_subgraph = self.wallet_graph.subgraph(wallet_sample)
        wallet_betweenness = nx.betweenness_centrality(wallet_subgraph, k=100)

        for node in wallet_sample:
            if wallet_degrees[node] >= wallet_p95:
                roles['wallet_graph'][node] = 'hub'
            elif wallet_degrees[node] == 1:
                roles['wallet_graph'][node] = 'leaf'
            elif wallet_betweenness.get(node, 0) > 0.01:
                roles['wallet_graph'][node] = 'bridge'
            else:
                roles['wallet_graph'][node] = 'intermediate'

        # Count role distributions
        role_counts = {
            'tx_graph': Counter(roles['tx_graph'].values()),
            'wallet_graph': Counter(roles['wallet_graph'].values())
        }

        return role_counts

    def compute_k_cores(self, max_k: int = 10):
        """Compute k-core decomposition"""

        results = {}

        # Transaction graph k-cores
        tx_core_numbers = nx.core_number(self.tx_graph.to_undirected())
        tx_core_distribution = Counter(tx_core_numbers.values())

        results['tx_graph'] = {
            'max_core': max(tx_core_numbers.values()),
            'core_distribution': dict(tx_core_distribution.most_common(max_k)),
            'nodes_in_max_core': sum(1 for v in tx_core_numbers.values()
                                     if v == max(tx_core_numbers.values()))
        }

        # Wallet graph k-cores (sample)
        wallet_sample = list(self.wallet_graph.nodes())[:10000]
        wallet_subgraph = self.wallet_graph.subgraph(wallet_sample).to_undirected()
        wallet_core_numbers = nx.core_number(wallet_subgraph)
        wallet_core_distribution = Counter(wallet_core_numbers.values())

        results['wallet_graph'] = {
            'max_core': max(wallet_core_numbers.values()) if wallet_core_numbers else 0,
            'core_distribution': dict(wallet_core_distribution.most_common(max_k)),
            'nodes_in_max_core': sum(1 for v in wallet_core_numbers.values()
                                     if v == max(wallet_core_numbers.values())) if wallet_core_numbers else 0
        }

        return results

    def _get_size_distribution(self, components: List):
        """Get distribution of component sizes"""
        sizes = [len(c) for c in components]
        return {
            'min': min(sizes),
            'max': max(sizes),
            'mean': np.mean(sizes),
            'median': np.median(sizes),
            'std': np.std(sizes),
            'num_singleton': sum(1 for s in sizes if s == 1),
            'num_large': sum(1 for s in sizes if s > 100)
        }

    def _compute_distribution_stats(self, values: List):
        """Compute statistics for a distribution"""
        if not values:
            return {}

        return {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p25': np.percentile(values, 25),
            'p75': np.percentile(values, 75),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }

    def _extract_bipartite_motifs(self, sample_size: int):
        """Extract motifs from bipartite graph"""

        # Sample transactions
        tx_nodes = [n for n in self.bipartite_graph.nodes() if n[0] == 'tx'][:sample_size]

        motifs = {
            'fan_in': [],  # Many wallets -> 1 tx
            'fan_out': [],  # 1 tx -> many wallets
            'pass_through': [],  # 1 wallet -> 1 tx -> 1 wallet
            'mixing': []  # Many wallets -> 1 tx -> many wallets
        }

        for tx_node in tx_nodes:
            input_wallets = [n for n in self.bipartite_graph.neighbors(tx_node)
                            if n[0] == 'wallet' and
                            self.bipartite_graph[tx_node][n].get('relation') == 'input']

            output_wallets = [n for n in self.bipartite_graph.neighbors(tx_node)
                             if n[0] == 'wallet' and
                             self.bipartite_graph[tx_node][n].get('relation') == 'output']

            num_inputs = len(input_wallets)
            num_outputs = len(output_wallets)

            if num_inputs > 3 and num_outputs == 1:
                motifs['fan_in'].append(tx_node)
            elif num_inputs == 1 and num_outputs > 3:
                motifs['fan_out'].append(tx_node)
            elif num_inputs == 1 and num_outputs == 1:
                motifs['pass_through'].append(tx_node)
            elif num_inputs > 3 and num_outputs > 3:
                motifs['mixing'].append(tx_node)

        return {
            'fan_in_count': len(motifs['fan_in']),
            'fan_out_count': len(motifs['fan_out']),
            'pass_through_count': len(motifs['pass_through']),
            'mixing_count': len(motifs['mixing'])
        }

    def save_results(self, output_dir: str = "results"):
        """Save analysis results to files"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        results = {
            'components': self.components,
            'structural_features': self.structural_features
        }

        with open(output_path / "structure_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)