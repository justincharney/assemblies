"""
Word Order Analysis - Manifold Comparison

Compares SVO vs OVS neural representations using geometric and topological
metrics from the manifold analysis pipeline.

Hypotheses to test:
1. OVS should show lower effective dimensionality (collapsed manifolds)
2. OVS should have fewer distinct H0 components
3. OVS should be less linear (more tangled representations)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from analyze_manifolds import analyze_geometry, analyze_topology
    from visualize_interactive import load_data, get_area_matrix_flat
except ImportError as e:
    print(f"Error: Could not import analysis modules: {e}")
    print("Make sure analyze_manifolds.py and visualize_interactive.py are in the same directory.")
    sys.exit(1)


def load_experiment_data(data_dir):
    """Load SVO and OVS experiment data."""
    svo_path = os.path.join(data_dir, "svo_data.npz")
    ovs_path = os.path.join(data_dir, "ovs_data.npz")

    data = {}

    if os.path.exists(svo_path):
        svo_npz = np.load(svo_path, allow_pickle=True)
        data['SVO'] = {key: svo_npz[key] for key in svo_npz.files}
        print(f"Loaded SVO data from {svo_path}")
    else:
        print(f"Warning: SVO data not found at {svo_path}")

    if os.path.exists(ovs_path):
        ovs_npz = np.load(ovs_path, allow_pickle=True)
        data['OVS'] = {key: ovs_npz[key] for key in ovs_npz.files}
        print(f"Loaded OVS data from {ovs_path}")
    else:
        print(f"Warning: OVS data not found at {ovs_path}")

    return data


def analyze_word_order_manifolds(data_dir, output_dir=None):
    """
    Compare manifold metrics between SVO and OVS.

    Returns DataFrame with metrics for each word order and area.
    """
    if output_dir is None:
        output_dir = data_dir

    os.makedirs(output_dir, exist_ok=True)

    # Key areas to analyze
    key_areas = [
        "ROLE_AGENT", "ROLE_ACTION", "ROLE_PATIENT",
        "SUBJ", "VERB_SYNTAX", "OBJ"
    ]

    results = []

    for word_order in ['SVO', 'OVS']:
        data_path = os.path.join(data_dir, f"{word_order.lower()}_data.npz")

        if not os.path.exists(data_path):
            print(f"Skipping {word_order} - data file not found")
            continue

        try:
            data, metadata, sentences = load_data(data_path)
        except Exception as e:
            print(f"Error loading {word_order} data: {e}")
            continue

        print(f"\n=== Analyzing {word_order} ===")

        for area in key_areas:
            if area not in metadata:
                print(f"  Skipping {area} - not in metadata")
                continue

            try:
                X, df = get_area_matrix_flat(data, area, sentences, num_trials=1)
            except Exception as e:
                print(f"  Error getting matrix for {area}: {e}")
                continue

            if X is None or len(X) == 0:
                print(f"  Skipping {area} - no data")
                continue

            print(f"  Analyzing {area} (shape: {X.shape})")

            row = {'word_order': word_order, 'area': area, 'n_points': X.shape[0]}

            # Geometric analysis
            try:
                geo_stats = analyze_geometry(X, f"{word_order}_{area}", output_dir)
                row.update(geo_stats)
            except Exception as e:
                print(f"    Geometry analysis failed: {e}")

            # Topological analysis
            if X.shape[0] >= 10:
                try:
                    topo_stats = analyze_topology(X, f"{word_order}_{area}", output_dir, save_plot=False)
                    row.update(topo_stats)
                except Exception as e:
                    print(f"    Topology analysis failed: {e}")

            results.append(row)

    return pd.DataFrame(results)


def compare_word_orders(df, output_dir):
    """Generate comparison plots and summary."""

    if df.empty:
        print("No data to compare")
        return

    # Metrics to compare
    key_metrics = [
        'Effective Dimensionality (PR)',
        'Dim (95% Var)',
        'Linearity (Geo/Euc Corr)',
        'H0 (Components) Feature Count'
    ]

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, metric in enumerate(key_metrics):
        if metric not in df.columns:
            continue

        ax = axes[i]

        # Pivot for comparison
        pivot = df.pivot(index='area', columns='word_order', values=metric)

        if pivot.empty:
            continue

        # Bar plot
        x = np.arange(len(pivot.index))
        width = 0.35

        if 'SVO' in pivot.columns:
            ax.bar(x - width/2, pivot['SVO'], width, label='SVO', color='steelblue')
        if 'OVS' in pivot.columns:
            ax.bar(x + width/2, pivot['OVS'], width, label='OVS', color='coral')

        ax.set_xlabel('Brain Area')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "word_order_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved to {plot_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("MANIFOLD METRICS COMPARISON")
    print("=" * 60)

    for metric in key_metrics:
        if metric not in df.columns:
            continue

        print(f"\n{metric}:")
        for word_order in ['SVO', 'OVS']:
            subset = df[df['word_order'] == word_order][metric]
            if not subset.empty:
                print(f"  {word_order}: mean={subset.mean():.3f}, std={subset.std():.3f}")

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze word order experiment results')
    parser.add_argument('--data-dir', default='word_order_experiment_results',
                        help='Directory with experiment data')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as data-dir)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir

    print("Word Order Manifold Analysis")
    print("=" * 60)

    # Analyze manifolds
    df = analyze_word_order_manifolds(args.data_dir, output_dir)

    if not df.empty:
        # Save metrics CSV
        csv_path = os.path.join(output_dir, "manifold_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to {csv_path}")

        # Generate comparison
        compare_word_orders(df, output_dir)

        # Print hypothesis evaluation
        print("\n" + "=" * 60)
        print("HYPOTHESIS EVALUATION")
        print("=" * 60)

        if 'Effective Dimensionality (PR)' in df.columns:
            svo_dim = df[df['word_order'] == 'SVO']['Effective Dimensionality (PR)'].mean()
            ovs_dim = df[df['word_order'] == 'OVS']['Effective Dimensionality (PR)'].mean()
            if not np.isnan(svo_dim) and not np.isnan(ovs_dim):
                result = "SUPPORTED" if ovs_dim > svo_dim else "NOT SUPPORTED"
                print(f"\nH1: OVS has higher dimensionality than SVO")
                print(f"    SVO: {svo_dim:.2f}, OVS: {ovs_dim:.2f}")
                print(f"    Result: {result}")

        if 'H0 (Components) Feature Count' in df.columns:
            svo_h0 = df[df['word_order'] == 'SVO']['H0 (Components) Feature Count'].mean()
            ovs_h0 = df[df['word_order'] == 'OVS']['H0 (Components) Feature Count'].mean()
            if not np.isnan(svo_h0) and not np.isnan(ovs_h0):
                result = "SUPPORTED" if ovs_h0 > svo_h0 else "NOT SUPPORTED"
                print(f"\nH2: OVS has more H0 components than SVO")
                print(f"    SVO: {svo_h0:.2f}, OVS: {ovs_h0:.2f}")
                print(f"    Result: {result}")

        if 'Linearity (Geo/Euc Corr)' in df.columns:
            svo_lin = df[df['word_order'] == 'SVO']['Linearity (Geo/Euc Corr)'].mean()
            ovs_lin = df[df['word_order'] == 'OVS']['Linearity (Geo/Euc Corr)'].mean()
            if not np.isnan(svo_lin) and not np.isnan(ovs_lin):
                result = "SUPPORTED" if ovs_lin < svo_lin else "NOT SUPPORTED"
                print(f"\nH3: OVS has lower linearity than SVO")
                print(f"    SVO: {svo_lin:.2f}, OVS: {ovs_lin:.2f}")
                print(f"    Result: {result}")

    else:
        print("No data available for analysis.")
        print("Run the experiment first: python run_word_order_difficulty_experiment.py")


if __name__ == "__main__":
    main()
