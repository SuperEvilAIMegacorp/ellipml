"""
Main analysis script for Bitcoin network structure analysis
Executes Phase 1-3: Structure Discovery, Tagging, and Visualization
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from structure_discovery import StructuralAnalyzer
from tagging_engine import TaggingEngine
from visualization_engine import VisualizationEngine


def run_phase1_structure_discovery():
    """Execute Phase 1: Structure Discovery"""

    print("\n" + "="*60)
    print("PHASE 1: STRUCTURE DISCOVERY")
    print("="*60)

    analyzer = StructuralAnalyzer(data_dir=".")

    # Load graphs
    print("\nLoading graph structures...")
    graph_stats = analyzer.load_graphs()
    print(f"Graphs loaded: {graph_stats['tx_nodes']:,} transactions, {graph_stats['wallet_nodes']:,} wallets")

    # Analyze components
    print("\nAnalyzing connected components...")
    components = analyzer.analyze_components()

    # Compute degree distributions
    print("\nComputing degree distributions...")
    degree_stats = analyzer.compute_degree_distributions()

    # Extract motifs
    print("\nExtracting structural motifs...")
    motifs = analyzer.extract_motifs(sample_size=500)

    # Detect roles
    print("\nDetecting structural roles...")
    roles = analyzer.detect_structural_roles()

    # K-core decomposition
    print("\nComputing k-core decomposition...")
    kcores = analyzer.compute_k_cores()

    results = {
        'graph_stats': graph_stats,
        'components': components,
        'degree_distributions': degree_stats,
        'motifs': motifs,
        'roles': roles,
        'kcores': kcores
    }

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "phase1_structure.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nPhase 1 results saved to {output_dir / 'phase1_structure.json'}")

    return analyzer, results


def run_phase2_tagging(analyzer):
    """Execute Phase 2: Tagging Strategy"""

    print("\n" + "="*60)
    print("PHASE 2: TAGGING STRATEGY")
    print("="*60)

    tagger = TaggingEngine(data_dir=".")

    # Load features and labels
    print("\nLoading features and labels...")
    data_stats = tagger.load_features_and_labels()
    print(f"Loaded {data_stats['tx_count']:,} transactions, {data_stats['wallet_count']:,} wallets")

    # Generate structural tags
    print("\nGenerating structural tags...")
    tx_sample = list(analyzer.tx_graph.nodes())[:2000]
    structural_tags_tx = tagger.generate_structural_tags(analyzer.tx_graph, tx_sample)

    wallet_sample = list(analyzer.wallet_graph.nodes())[:2000]
    structural_tags_wallet = tagger.generate_structural_tags(analyzer.wallet_graph, wallet_sample)

    tagger.tags['structural'] = {**structural_tags_tx, **structural_tags_wallet}

    # Generate behavioral tags
    print("\nGenerating behavioral tags...")
    behavioral_tags = tagger.generate_behavioral_tags()
    tagger.tags['behavioral'] = behavioral_tags

    # Generate temporal tags
    print("\nGenerating temporal tags...")
    temporal_tags = tagger.create_temporal_tags()
    tagger.tags['temporal'] = temporal_tags

    # Generate propagated tags (expensive, so using smaller sample)
    print("\nGenerating propagated tags...")

    # Prepare labels for propagation
    tx_labels = tagger.tx_labels.set_index('txId')['class'].to_dict()
    tx_subgraph = analyzer.tx_graph.subgraph(list(analyzer.tx_graph.nodes())[:2000])
    propagated_tags = tagger.generate_propagated_tags(tx_subgraph, tx_labels, max_hops=1)
    tagger.tags['propagated'] = propagated_tags

    # Generate composite tags
    print("\nGenerating composite tags...")
    composite_tags = tagger.generate_composite_tags()

    # Save tags
    output_dir = Path("results")
    tagger.save_tags(str(output_dir))

    # Create summary statistics
    tag_stats = {
        'structural_tags_count': len(tagger.tags.get('structural', {})),
        'behavioral_tags_count': len(tagger.tags.get('behavioral', {})),
        'temporal_tags_count': len(tagger.tags.get('temporal', {})),
        'propagated_tags_count': len(tagger.tags.get('propagated', {})),
        'composite_patterns_count': len(composite_tags)
    }

    with open(output_dir / "phase2_tagging.json", "w") as f:
        json.dump(tag_stats, f, indent=2)

    print(f"\nPhase 2 results saved to {output_dir / 'phase2_tagging.json'}")
    print(f"Tag details saved to {output_dir / 'node_tags.csv'}")

    return tagger, tag_stats


def run_phase3_visualization(analyzer, tagger, structure_results, tag_stats):
    """Execute Phase 3: Visualization"""

    print("\n" + "="*60)
    print("PHASE 3: VISUALIZATION")
    print("="*60)

    viz = VisualizationEngine()
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)

    # Plot degree distributions
    print("\nCreating degree distribution plots...")
    viz.plot_degree_distributions(
        structure_results['degree_distributions'],
        output_path=str(output_dir / "degree_distributions.png")
    )

    # Plot component analysis
    print("\nCreating component analysis plots...")
    viz.plot_component_analysis(
        structure_results['components'],
        output_path=str(output_dir / "component_analysis.png")
    )

    # Plot motif analysis
    print("\nCreating motif analysis plots...")
    viz.plot_motif_analysis(
        structure_results['motifs'],
        output_path=str(output_dir / "motif_patterns.png")
    )

    # Plot role distribution
    print("\nCreating role distribution plots...")
    viz.plot_role_distribution(
        structure_results['roles'],
        output_path=str(output_dir / "role_distribution.png")
    )

    # Plot tag distribution
    print("\nCreating tag distribution plots...")
    for tag_type in ['structural', 'behavioral', 'temporal']:
        if tag_type in tagger.tags and tagger.tags[tag_type]:
            viz.plot_tag_distribution(
                tagger.tags,
                tag_type=tag_type,
                output_path=str(output_dir / f"tag_distribution_{tag_type}.png")
            )

    # Sample ego networks
    print("\nCreating ego network visualizations...")

    # Get some sample nodes with different roles
    sample_nodes = list(analyzer.tx_graph.nodes())[:5]
    labels = tagger.tx_labels.set_index('txId')['class'].to_dict()

    for idx, node in enumerate(sample_nodes):
        viz.plot_ego_network(
            analyzer.tx_graph,
            node,
            max_hops=2,
            labels=labels,
            output_path=str(output_dir / f"ego_network_{idx}.png")
        )

    # Create temporal heatmap
    print("\nCreating temporal activity heatmap...")
    if tagger.wallet_features is not None:
        viz.plot_temporal_heatmap(
            tagger.wallet_features,
            sample_size=100,
            output_path=str(output_dir / "temporal_heatmap.png")
        )

    # Create summary dashboard
    print("\nCreating summary dashboard...")
    summary_stats = {
        **structure_results['graph_stats'],
        'tx_wcc': structure_results['components']['tx_graph']['num_weakly_connected'],
        'tx_scc': structure_results['components']['tx_graph']['num_strongly_connected'],
        'wallet_wcc': structure_results['components']['wallet_graph']['num_weakly_connected'],
        'wallet_scc': structure_results['components']['wallet_graph']['num_strongly_connected'],
        'linear_chains': structure_results['motifs']['transaction_motifs']['linear_chain_count'],
        'forks': structure_results['motifs']['transaction_motifs']['fork_count'],
        'merges': structure_results['motifs']['transaction_motifs']['merge_count'],
        'pass_through': structure_results['motifs']['bipartite_motifs']['pass_through_count'],
        'hubs': structure_results['roles']['tx_graph'].get('hub', 0),
        'leaves': structure_results['roles']['tx_graph'].get('leaf', 0),
        'bridges': structure_results['roles']['tx_graph'].get('bridge', 0),
        'intermediate': structure_results['roles']['tx_graph'].get('intermediate', 0),
        **tag_stats
    }

    viz.create_summary_dashboard(
        summary_stats,
        output_path=str(output_dir / "summary_dashboard.png")
    )

    print(f"\nVisualization results saved to {output_dir}/")


def main():
    """Main execution function"""

    print("\n" + "="*60)
    print("BITCOIN NETWORK STRUCTURE ANALYSIS")
    print(f"Execution started: {datetime.now()}")
    print("="*60)

    # Phase 1: Structure Discovery
    analyzer, structure_results = run_phase1_structure_discovery()

    # Phase 2: Tagging
    tagger, tag_stats = run_phase2_tagging(analyzer)

    # Phase 3: Visualization
    run_phase3_visualization(analyzer, tagger, structure_results, tag_stats)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Execution finished: {datetime.now()}")
    print("="*60)

    # Final summary
    print("\nSummary of Results:")
    print(f"  - Structure analysis: {structure_results['graph_stats']['tx_nodes']:,} transactions analyzed")
    print(f"  - Components found: TX={structure_results['components']['tx_graph']['num_weakly_connected']}, "
          f"Wallet={structure_results['components']['wallet_graph']['num_weakly_connected']}")
    print(f"  - Motifs extracted: {sum(structure_results['motifs']['transaction_motifs'].values())}")
    print(f"  - Tags generated: {sum(tag_stats.values())}")
    print(f"  - Visualizations created: 10+ plots")

    print("\nOutput directories:")
    print("  - results/: JSON analysis results and CSV tags")
    print("  - visualizations/: All generated plots")


if __name__ == "__main__":
    main()