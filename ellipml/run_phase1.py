"""
Run Phase 1 analysis only
"""

import json
import time
from pathlib import Path
from structure_discovery import StructuralAnalyzer

def main():
    print("\n" + "="*60)
    print("PHASE 1: STRUCTURE DISCOVERY (Simplified)")
    print("="*60)

    analyzer = StructuralAnalyzer(data_dir=".")

    # Load graphs
    print("\nStep 1: Loading graph structures...")
    start = time.time()
    graph_stats = analyzer.load_graphs()
    print(f"  Completed in {time.time() - start:.2f}s")
    print(f"  Loaded: {graph_stats['tx_nodes']:,} tx nodes, {graph_stats['wallet_nodes']:,} wallet nodes")

    # Analyze components
    print("\nStep 2: Analyzing connected components...")
    start = time.time()
    components = analyzer.analyze_components()
    print(f"  Completed in {time.time() - start:.2f}s")
    print(f"  TX components: {components['tx_graph']['num_weakly_connected']}")
    print(f"  Wallet components: {components['wallet_graph']['num_weakly_connected']}")

    # Compute degree distributions
    print("\nStep 3: Computing degree distributions...")
    start = time.time()
    degree_stats = analyzer.compute_degree_distributions()
    print(f"  Completed in {time.time() - start:.2f}s")
    print(f"  TX mean degree: {degree_stats['tx_graph']['total_degree']['mean']:.2f}")
    print(f"  Wallet mean degree: {degree_stats['wallet_graph']['total_degree']['mean']:.2f}")

    # Extract motifs (small sample)
    print("\nStep 4: Extracting structural motifs (sample=100)...")
    start = time.time()
    motifs = analyzer.extract_motifs(sample_size=100)
    print(f"  Completed in {time.time() - start:.2f}s")
    print(f"  Linear chains: {motifs['transaction_motifs']['linear_chain_count']}")
    print(f"  Forks: {motifs['transaction_motifs']['fork_count']}")

    # Detect roles (small sample)
    print("\nStep 5: Detecting structural roles (sample)...")
    start = time.time()
    roles = analyzer.detect_structural_roles()
    print(f"  Completed in {time.time() - start:.2f}s")
    print(f"  TX roles: {dict(roles['tx_graph'])}")

    # Save results
    results = {
        'graph_stats': graph_stats,
        'components': components,
        'degree_distributions': degree_stats,
        'motifs': motifs,
        'roles': {k: dict(v) for k, v in roles.items()}
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "phase1_structure.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Phase 1 results saved to {output_dir / 'phase1_structure.json'}")
    print("="*60)

    return results

if __name__ == "__main__":
    results = main()