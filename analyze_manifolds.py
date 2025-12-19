import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from ripser import ripser
from persim import plot_diagrams
import warnings

# Import data loading from the existing visualization script
try:
    from visualize_interactive import load_data, get_area_matrix_flat
except ImportError:
    print("Error: Could not import 'visualize_interactive.py'. Make sure it is in the same directory.")
    sys.exit(1)

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def analyze_topology(X, area_name, output_dir):
    """
    Performs Topological Data Analysis (TDA) using Persistent Homology.
    Computes Betti numbers and plots Persistence Diagrams.
    """
    print(f"  [Topology] Computing persistent homology (Vietoris-Rips)...")
    
    # Subsample if too large (Ripser is O(N^3) roughly)
    max_points = 1000
    if X.shape[0] > max_points:
        print(f"    Subsampling from {X.shape[0]} to {max_points} points for TDA efficiency.")
        indices = np.random.choice(X.shape[0], max_points, replace=False)
        X_tda = X[indices]
    else:
        X_tda = X

    # Compute persistence diagrams
    # maxdim=2 computes H0 (components), H1 (loops), H2 (voids)
    try:
        result = ripser(X_tda, maxdim=2)
        diagrams = result['dgms']
    except Exception as e:
        print(f"    Error in Ripser: {e}")
        return {}

    # Plot and save diagrams
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_diagrams(diagrams, show=False)
    plt.title(f"{area_name} Persistence")
    
    # Plot barcodes (lifetimes)
    plt.subplot(1, 2, 2)
    plot_diagrams(diagrams, show=False, lifetime=True)
    plt.title(f"{area_name} Lifetimes")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{area_name}_topology.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Calculate statistics
    stats = {}
    dims = ["H0 (Components)", "H1 (Loops)", "H2 (Voids)"]
    
    for i, dgm in enumerate(diagrams):
        if i >= len(dims): break
        
        # Lifetime = Death - Birth
        # Filter out infinite death (usually one connected component in H0)
        lifetimes = dgm[:, 1] - dgm[:, 0]
        finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
        
        if len(finite_lifetimes) > 0:
            avg_life = np.mean(finite_lifetimes)
            max_life = np.max(finite_lifetimes)
            sum_life = np.sum(finite_lifetimes) # Total persistence
        else:
            avg_life = 0
            max_life = 0
            sum_life = 0
            
        stats[f"{dims[i]} Max Lifetime"] = max_life
        stats[f"{dims[i]} Avg Lifetime"] = avg_life
        stats[f"{dims[i]} Total Persistence"] = sum_life
        stats[f"{dims[i]} Feature Count"] = len(dgm)

    return stats

def analyze_geometry(X, area_name, output_dir):
    """
    Computes geometric signatures:
    1. Intrinsic Dimensionality (PCA participation ratio)
    2. Linearity (Geodesic vs Euclidean correlation)
    3. Local Extrinsic Curvature (Residual variance of local PCA)
    """
    print(f"  [Geometry] Computing geometric signatures...")
    stats = {}
    
    # 1. Effective Dimensionality (Participation Ratio of PCA eigenvalues)
    pca = PCA()
    pca.fit(X)
    expl_var = pca.explained_variance_ratio_
    # Participation Ratio = (Sum lambda)^2 / Sum (lambda^2)
    # Since sum(expl_var) = 1, this is 1 / sum(expl_var^2)
    pr_dim = 1.0 / np.sum(expl_var**2)
    stats["Effective Dimensionality (PR)"] = pr_dim
    
    # Also 95% variance dim
    cumsum_var = np.cumsum(expl_var)
    dim_95 = np.argmax(cumsum_var >= 0.95) + 1
    stats["Dim (95% Var)"] = dim_95
    
    # 2. Geodesic vs Euclidean (Global curvature/non-linearity)
    # Construct k-NN graph
    k = 10
    if X.shape[0] < k + 1:
        k = X.shape[0] - 1
        
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    adj_matrix = nbrs.kneighbors_graph(X, mode='distance')
    
    # Compute shortest paths (Geodesic approximation)
    # Using unweighted or distance-weighted graph. Isomap uses distance-weighted.
    geodesic_dist = shortest_path(adj_matrix, directed=False)
    
    # Euclidean distance matrix
    euclidean_dist = squareform(pdist(X))
    
    # Compare upper triangles
    mask = np.triu(np.ones(geodesic_dist.shape, dtype=bool), k=1)
    geo_vals = geodesic_dist[mask]
    euc_vals = euclidean_dist[mask]
    
    # Handle disconnected components (infinite geodesic distance)
    finite_mask = np.isfinite(geo_vals)
    if np.sum(finite_mask) > 10:
        correlation = np.corrcoef(geo_vals[finite_mask], euc_vals[finite_mask])[0, 1]
        stats["Linearity (Geo/Euc Corr)"] = correlation
        stats["Max Geodesic Dist (Diameter)"] = np.max(geo_vals[finite_mask])
    else:
        stats["Linearity (Geo/Euc Corr)"] = np.nan
        stats["Max Geodesic Dist (Diameter)"] = np.nan

    # 3. Local Extrinsic Curvature
    # Estimate using residual variance of local PCA on k-neighbors
    # We assume the manifold is locally 'dim_95' dimensional or smaller (e.g., 2D or 3D).
    # We'll calculate the residual variance not explained by the top 3 components 
    # (assuming we are looking for curvature in embedding space).
    local_curvatures = []
    local_dim = 3 # Hardcoded assumption for "low dimensional manifold"
    
    # Indices of neighbors
    _, indices = nbrs.kneighbors(X)
    
    for i in range(X.shape[0]):
        local_pts = X[indices[i]]
        # Center
        local_pts = local_pts - np.mean(local_pts, axis=0)
        # SVD
        # Handling small k
        if local_pts.shape[0] > local_dim:
            _, s, _ = np.linalg.svd(local_pts)
            # Total variance
            total_var = np.sum(s**2)
            if total_var > 1e-9:
                # Residual variance (sum of squared singular values after top local_dim)
                residual_var = np.sum(s[local_dim:]**2)
                local_curvatures.append(residual_var / total_var)
            else:
                local_curvatures.append(0.0)
        else:
             local_curvatures.append(0.0)

    stats["Avg Local Curvature (Res. Var)"] = np.mean(local_curvatures)
    
    return stats

def analyze_separation(data, area_name, sentences, num_trials=2):
    """
    Computes the separation (distance) between the manifolds of different sentences.
    Uses a modified Hausdorff distance (mean of min distances) to be robust to outliers.
    """
    print(f"  [Separation] Computing inter-sentence distances...")
    
    # Extract point cloud for each sentence
    sentence_clouds = {}
    valid_sentences = []
    
    for s_idx, sent in enumerate(sentences):
        # Collect all points for this sentence across trials
        pts = []
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            if key in data:
                for winners in data[key]:
                    # We need the high-dim representation (one-hot)
                    # To save memory, we'll reconstruct it on the fly or just use indices
                    # But wait, we need the *same* feature space as X. 
                    # The 'get_area_matrix_flat' function did the mapping. 
                    # Let's reuse the mapping logic or pass the pre-computed X with labels.
                    pass 
    
    # Better approach: Use the dataframe 'df' and matrix 'X' we already have
    # We need to map X rows back to sentences
    return {}

def analyze_separation_from_X(X, df):
    """
    Computes pairwise distances between sentence clusters using the already flattened X and metadata df.
    """
    print(f"  [Separation] Computing inter-sentence distances...")
    
    unique_sentences = df["Sentence"].unique()
    n_sent = len(unique_sentences)
    
    if n_sent < 2:
        return {"Avg Inter-Sentence Dist": 0.0}
    
    # Pre-group indices by sentence
    sent_indices = {sent: df.index[df["Sentence"] == sent].tolist() for sent in unique_sentences}
    
    distances = []
    
    # Compute pairwise distances between sentence manifolds
    # We'll use a subset to save time if n_sent is large
    import itertools
    pairs = list(itertools.combinations(unique_sentences, 2))
    
    # Subsample pairs if too many
    if len(pairs) > 100:
        pairs = [pairs[i] for i in np.random.choice(len(pairs), 100, replace=False)]
        
    for s1, s2 in pairs:
        X1 = X[sent_indices[s1]]
        X2 = X[sent_indices[s2]]
        
        # Hausdorff-like metric: Mean of minimum distances (Chamfer distance)
        # d(A, B) = 1/2 * (mean(min(dist(a, B))) + mean(min(dist(b, A))))
        
        from sklearn.metrics import pairwise_distances
        d_mat = pairwise_distances(X1, X2)
        
        d1 = np.mean(np.min(d_mat, axis=1)) # Avg dist from A to nearest in B
        d2 = np.mean(np.min(d_mat, axis=0)) # Avg dist from B to nearest in A
        
        dist = (d1 + d2) / 2.0
        distances.append(dist)
        
    return {
        "Avg Inter-Sentence Dist": np.mean(distances),
        "Std Inter-Sentence Dist": np.std(distances),
        "Min Inter-Sentence Dist": np.min(distances),
        "Max Inter-Sentence Dist": np.max(distances)
    }

def main():
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "learner_data.npz"

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "manifold_analysis"

    os.makedirs(output_dir, exist_ok=True)
    
    data, metadata, sentences = load_data(data_file)
    
    # Analyze all available areas in the data
    areas = list(metadata.keys())
    
    results = []

    for area in areas:
        if area not in metadata: continue
        
        print(f"\nAnalyzing Area: {area}")
        # num_trials=2 to match visualization default, captures variability
        X, df = get_area_matrix_flat(data, area, sentences, num_trials=2)
        
        if X is None or len(X) == 0:
            print(f"  Skipping {area} (No data)")
            continue
        
        print(f"  Data Shape: {X.shape}")
        
        # Run analyses
        geo_stats = analyze_geometry(X, area, output_dir)
        topo_stats = analyze_topology(X, area, output_dir)
        sep_stats = analyze_separation_from_X(X, df)
        
        # Combine
        combined = {"Area": area}
        combined.update(geo_stats)
        combined.update(topo_stats)
        combined.update(sep_stats)
        results.append(combined)

    # Save summary CSV
    if results:
        df_res = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "manifold_metrics.csv")
        df_res.to_csv(csv_path, index=False)
        print(f"\nAnalysis complete. Metrics saved to {csv_path}")
        print("\nSummary Table:")
        print(df_res.to_string())
        print(f"\nPlots saved in {output_dir}/")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
