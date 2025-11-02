"""
Phase 2: Tagging Strategy
Create structural, behavioral, and propagated tags for nodes
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Set, List, Optional, Tuple
from pathlib import Path


class TaggingEngine:
    """Generate multi-dimensional tags for nodes based on structure and behavior"""

    def __init__(self, data_dir: str = ".."):
        self.data_dir = Path(data_dir)
        self.tx_features = None
        self.wallet_features = None
        self.tx_labels = None
        self.wallet_labels = None
        self.tags = {
            'structural': {},
            'behavioral': {},
            'propagated': {}
        }

    def load_features_and_labels(self):
        """Load node features and labels"""

        # Load transaction features and labels
        self.tx_features = pd.read_csv(
            self.data_dir / "ellipticpp/txs_features.csv",
            header=None
        )
        self.tx_labels = pd.read_csv(
            self.data_dir / "ellipticpp/txs_classes.csv"
        )

        # Load wallet features and labels
        self.wallet_features = pd.read_csv(
            self.data_dir / "ellipticpp/wallets_features_classes_combined.csv"
        )
        self.wallet_labels = pd.read_csv(
            self.data_dir / "ellipticpp/wallets_classes.csv"
        )

        return {
            'tx_count': len(self.tx_features),
            'wallet_count': len(self.wallet_features['address'].unique()),
            'tx_labeled': self.tx_labels['class'].notna().sum(),
            'wallet_labeled': self.wallet_labels['class'].notna().sum()
        }

    def generate_structural_tags(self, graph: nx.Graph, node_sample: Optional[List] = None):
        """Generate tags based on graph structure"""

        if node_sample is None:
            node_sample = list(graph.nodes())[:10000]

        structural_tags = {}

        # Compute degree-based tags
        degrees = dict(graph.degree(node_sample))
        degree_values = list(degrees.values())

        if degree_values:
            p25 = np.percentile(degree_values, 25)
            p75 = np.percentile(degree_values, 75)
            p95 = np.percentile(degree_values, 95)

            for node in node_sample:
                deg = degrees[node]
                tags = []

                # Degree categories
                if deg == 0:
                    tags.append('isolated')
                elif deg == 1:
                    tags.append('leaf')
                elif deg >= p95:
                    tags.append('hub')
                elif deg >= p75:
                    tags.append('high_degree')
                elif deg <= p25:
                    tags.append('low_degree')
                else:
                    tags.append('medium_degree')

                # In/out degree ratio for directed graphs
                if graph.is_directed():
                    in_deg = graph.in_degree(node)
                    out_deg = graph.out_degree(node)

                    if in_deg > 0 and out_deg == 0:
                        tags.append('sink')
                    elif out_deg > 0 and in_deg == 0:
                        tags.append('source')
                    elif in_deg > out_deg * 2:
                        tags.append('collector')
                    elif out_deg > in_deg * 2:
                        tags.append('distributor')
                    else:
                        tags.append('balanced')

                structural_tags[node] = tags

        # K-core tags
        if len(node_sample) < 50000:  # Only for smaller samples
            try:
                core_numbers = nx.core_number(graph.to_undirected())
                max_core = max(core_numbers.values()) if core_numbers else 0

                for node in node_sample:
                    if node in core_numbers:
                        core = core_numbers[node]
                        if core == max_core and max_core > 2:
                            structural_tags[node].append('max_core')
                        elif core >= max_core * 0.8:
                            structural_tags[node].append('high_core')
                        elif core <= 2:
                            structural_tags[node].append('periphery')
            except:
                pass  # Skip if graph is too disconnected

        # Clustering coefficient
        if len(node_sample) < 10000:
            clustering = nx.clustering(graph.to_undirected(), nodes=node_sample)
            for node in node_sample:
                if node in clustering:
                    if clustering[node] > 0.5:
                        structural_tags[node].append('highly_clustered')
                    elif clustering[node] == 0:
                        structural_tags[node].append('non_clustered')

        return structural_tags

    def generate_behavioral_tags(self):
        """Generate tags based on node behavioral features"""

        behavioral_tags = {}

        # Transaction behavioral tags
        if self.tx_features is not None:
            tx_ids = self.tx_features[0].values  # First column is transaction ID
            timesteps = self.tx_features[1].values  # Second column is timestep

            # Analyze feature distributions for outliers
            feature_matrix = self.tx_features.iloc[:, 2:].values  # Skip ID and timestep
            feature_means = np.mean(feature_matrix, axis=0)
            feature_stds = np.std(feature_matrix, axis=0)

            for idx, tx_id in enumerate(tx_ids[:10000]):  # Sample for efficiency
                tags = []

                # Temporal tags
                ts = timesteps[idx]
                if ts <= 10:
                    tags.append('early_period')
                elif ts >= 40:
                    tags.append('late_period')
                else:
                    tags.append('middle_period')

                # Feature anomaly detection (simple z-score)
                tx_features = feature_matrix[idx]
                z_scores = np.abs((tx_features - feature_means) / (feature_stds + 1e-8))
                if np.any(z_scores > 3):
                    tags.append('feature_outlier')

                behavioral_tags[('tx', tx_id)] = tags

        # Wallet behavioral tags
        if self.wallet_features is not None:
            # Sample wallets for analysis
            wallet_sample = self.wallet_features.drop_duplicates(subset='address').head(10000)

            for _, wallet in wallet_sample.iterrows():
                tags = []
                addr = wallet['address']

                # Activity level tags
                total_txs = wallet.get('total_txs', 0)
                if total_txs > wallet_sample['total_txs'].quantile(0.9):
                    tags.append('high_activity')
                elif total_txs < wallet_sample['total_txs'].quantile(0.1):
                    tags.append('low_activity')

                # Lifetime tags
                lifetime = wallet.get('lifetime_in_blocks', 0)
                if lifetime > wallet_sample['lifetime_in_blocks'].quantile(0.9):
                    tags.append('long_lived')
                elif lifetime < wallet_sample['lifetime_in_blocks'].quantile(0.1):
                    tags.append('short_lived')

                # Volume tags
                btc_total = wallet.get('btc_transacted_total', 0)
                if btc_total > wallet_sample['btc_transacted_total'].quantile(0.9):
                    tags.append('high_volume')
                elif btc_total < wallet_sample['btc_transacted_total'].quantile(0.1):
                    tags.append('low_volume')

                # Direction bias
                sender_txs = wallet.get('num_txs_as_sender', 0)
                receiver_txs = wallet.get('num_txs_as receiver', 0)

                if sender_txs > 0 and receiver_txs == 0:
                    tags.append('sender_only')
                elif receiver_txs > 0 and sender_txs == 0:
                    tags.append('receiver_only')
                elif sender_txs > receiver_txs * 2:
                    tags.append('primarily_sender')
                elif receiver_txs > sender_txs * 2:
                    tags.append('primarily_receiver')

                behavioral_tags[('wallet', addr)] = tags

        return behavioral_tags

    def generate_propagated_tags(self, graph: nx.Graph, labels: Dict,
                                 max_hops: int = 2):
        """Generate tags based on label propagation from neighbors"""

        propagated_tags = {}

        # Get labeled nodes
        illicit_nodes = {n for n, l in labels.items() if l == '1' or l == 1}
        licit_nodes = {n for n, l in labels.items() if l == '2' or l == 2}
        unknown_nodes = {n for n in graph.nodes()
                        if n not in illicit_nodes and n not in licit_nodes}

        # Sample for efficiency
        sample_size = min(5000, len(unknown_nodes))
        unknown_sample = list(unknown_nodes)[:sample_size]

        for node in unknown_sample:
            tags = []

            # 1-hop exposure
            neighbors_1hop = set(graph.neighbors(node))
            illicit_1hop = len(neighbors_1hop & illicit_nodes)
            licit_1hop = len(neighbors_1hop & licit_nodes)
            total_1hop = len(neighbors_1hop)

            if total_1hop > 0:
                illicit_ratio_1hop = illicit_1hop / total_1hop
                licit_ratio_1hop = licit_1hop / total_1hop

                if illicit_ratio_1hop > 0.8:
                    tags.append('strong_illicit_exposure_1hop')
                elif illicit_ratio_1hop > 0.3:
                    tags.append('moderate_illicit_exposure_1hop')
                elif illicit_ratio_1hop > 0:
                    tags.append('weak_illicit_exposure_1hop')

                if licit_ratio_1hop > 0.8:
                    tags.append('strong_licit_exposure_1hop')

            # 2-hop exposure (expensive, so only for smaller samples)
            if len(unknown_sample) < 1000 and max_hops >= 2:
                neighbors_2hop = set()
                for n1 in neighbors_1hop:
                    neighbors_2hop.update(graph.neighbors(n1))
                neighbors_2hop -= {node}  # Remove self

                illicit_2hop = len(neighbors_2hop & illicit_nodes)
                licit_2hop = len(neighbors_2hop & licit_nodes)
                total_2hop = len(neighbors_2hop)

                if total_2hop > 0:
                    illicit_ratio_2hop = illicit_2hop / total_2hop

                    if illicit_ratio_2hop > 0.5:
                        tags.append('illicit_exposure_2hop')
                    if licit_2hop / total_2hop > 0.5:
                        tags.append('licit_exposure_2hop')

            # Isolation tags
            if illicit_1hop == 0 and total_1hop > 0:
                tags.append('no_direct_illicit_contact')
            if illicit_1hop == 0 and illicit_2hop == 0:
                tags.append('isolated_from_illicit')

            propagated_tags[node] = tags

        return propagated_tags

    def create_temporal_tags(self):
        """Create tags based on temporal patterns"""

        temporal_tags = {}

        if self.wallet_features is not None:
            # Analyze wallet temporal patterns
            wallet_sample = self.wallet_features.drop_duplicates(subset='address').head(10000)

            for _, wallet in wallet_sample.iterrows():
                tags = []
                addr = wallet['address']

                # Multi-timestep activity
                num_timesteps = wallet.get('num_timesteps_appeared_in', 1)
                if num_timesteps == 1:
                    tags.append('single_timestep')
                elif num_timesteps > 5:
                    tags.append('persistent')
                else:
                    tags.append('multi_timestep')

                # Activity concentration
                total_txs = wallet.get('total_txs', 1)
                if total_txs > 0:
                    concentration = num_timesteps / total_txs
                    if concentration < 0.1:
                        tags.append('burst_activity')
                    elif concentration > 0.5:
                        tags.append('sparse_activity')

                # Blocks between transactions
                blocks_mean = wallet.get('blocks_btwn_txs_mean', 0)
                if blocks_mean > 0:
                    if blocks_mean < 100:
                        tags.append('rapid_transactions')
                    elif blocks_mean > 10000:
                        tags.append('slow_transactions')

                temporal_tags[('wallet', addr)] = tags

        return temporal_tags

    def generate_composite_tags(self):
        """Combine different tag types to create composite tags"""

        composite_tags = {}

        # Merge all tag types for each node
        all_nodes = set()
        all_nodes.update(self.tags.get('structural', {}).keys())
        all_nodes.update(self.tags.get('behavioral', {}).keys())
        all_nodes.update(self.tags.get('propagated', {}).keys())

        for node in all_nodes:
            composite = set()

            # Collect all tags for this node
            if node in self.tags.get('structural', {}):
                composite.update(self.tags['structural'][node])
            if node in self.tags.get('behavioral', {}):
                composite.update(self.tags['behavioral'][node])
            if node in self.tags.get('propagated', {}):
                composite.update(self.tags['propagated'][node])

            # Generate composite patterns
            composite_patterns = []

            # High-risk patterns
            if 'hub' in composite and 'strong_illicit_exposure_1hop' in composite:
                composite_patterns.append('high_risk_hub')

            if 'short_lived' in composite and 'high_volume' in composite:
                composite_patterns.append('rapid_high_volume')

            if 'collector' in composite and 'single_timestep' in composite:
                composite_patterns.append('collection_point')

            if 'distributor' in composite and 'burst_activity' in composite:
                composite_patterns.append('distribution_burst')

            # Low-risk patterns
            if 'isolated_from_illicit' in composite and 'long_lived' in composite:
                composite_patterns.append('established_clean')

            if 'strong_licit_exposure_1hop' in composite and 'persistent' in composite:
                composite_patterns.append('stable_licit_network')

            composite_tags[node] = {
                'all_tags': list(composite),
                'composite_patterns': composite_patterns
            }

        return composite_tags

    def save_tags(self, output_dir: str = "tags"):
        """Save generated tags to files"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save as DataFrame for easy analysis
        tag_records = []

        for node, tags in self.tags.get('structural', {}).items():
            tag_records.append({
                'node': node,
                'tag_type': 'structural',
                'tags': ','.join(tags)
            })

        for node, tags in self.tags.get('behavioral', {}).items():
            tag_records.append({
                'node': node,
                'tag_type': 'behavioral',
                'tags': ','.join(tags)
            })

        for node, tags in self.tags.get('propagated', {}).items():
            tag_records.append({
                'node': node,
                'tag_type': 'propagated',
                'tags': ','.join(tags)
            })

        if tag_records:
            pd.DataFrame(tag_records).to_csv(output_path / "node_tags.csv", index=False)