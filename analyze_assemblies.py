import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dPCA import dPCA
import phate
import umap
import os
import sys

# Create output directory
os.makedirs("plots", exist_ok=True)

def load_data(filepath="assemblies_data.npz"):
    print(f"Loading {filepath}...")
    data = np.load(filepath, allow_pickle=True)
    metadata = data['metadata'].item()
    sentences = data['sentences']
    
    return data, metadata, sentences

def get_area_matrix(data, area_name, n_neurons, sentences, num_trials=2):
    # Collect all firing indices to build a reduced basis if n_neurons is large
    all_winners = set()
    
    # X: array-like of shape (n_trials, n_neurons, n_time_bins, n_conditions) 
    traces = []
    
    # We iterate to find union of active neurons first
    for s_idx in range(len(sentences)):
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            if key not in data:
                continue
            history = data[key] # object array of lists
            for step_winners in history:
                all_winners.update(step_winners)
                
    sorted_winners = sorted(list(all_winners))
    winner_map = {idx: i for i, idx in enumerate(sorted_winners)}
    reduced_n = len(sorted_winners)
    print(f"Area {area_name}: Original N={n_neurons}, Active N={reduced_n}")
    
    # Now construct matrices
    # We'll store a list of (time, reduced_n) matrices
    matrix_list = []
    labels_list = [] # (sentence_idx, trial_idx, time_step)
    
    expected_length = None

    for s_idx, sentence in enumerate(sentences):
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            if key not in data:
                # Should not happen
                continue
            time_len = len(data[key])
            if expected_length is None:
                expected_length = time_len
            elif time_len != expected_length:
                raise RuntimeError(
                    f"{area_name} has mismatched history lengths: expected {expected_length} steps "
                    f"but {key} has {time_len} for sentence '{sentence}'"
                )
    if expected_length is None:
        raise RuntimeError(f"No history found for area {area_name}; cannot build dPCA matrices")

    time_steps = expected_length

    # We will pad with zeros for dPCA compatibility if we want to include all sentences
    X_dpca = np.zeros((num_trials, reduced_n, time_steps, len(sentences)))
    # X_dpca shape: (n_trials, n_neurons, n_time, n_conditions)
    
    # For standard analysis (PCA/UMAP), just concatenation
    X_concat = []
    y_sentence = []
    y_time = []
    
    for s_idx, sentence in enumerate(sentences):
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            history = data[key]
            
            # Build dense matrix for this trial
            trial_mat = np.zeros((len(history), reduced_n))
            for t, winners in enumerate(history):
                indices = [winner_map[w] for w in winners if w in winner_map]
                trial_mat[t, indices] = 1.0
            
            X_concat.append(trial_mat)
            y_sentence.extend([s_idx] * len(history))
            y_time.extend(list(range(len(history))))
            
            # Fill dPCA matrix (equal lengths enforced above)
            X_dpca[t_idx, :, :, s_idx] = trial_mat.T

    X_concat = np.vstack(X_concat)
    y_sentence = np.array(y_sentence)
    y_time = np.array(y_time)
    
    return X_concat, y_sentence, y_time, X_dpca, sentences

def run_pca(X, y_s, y_t, area_name, sentences):
    print(f"Running PCA on {area_name}...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.jet(np.linspace(0, 1, len(sentences)))
    
    for s_idx in range(len(sentences)):
        mask = (y_s == s_idx)
        # We might have multiple trials, they will be plotted together
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                   c=colors[s_idx:s_idx+1], label=sentences[s_idx], s=5, alpha=0.5)
        
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.title(f"PCA - {area_name}")
    plt.savefig(f"plots/pca_{area_name}.png")
    plt.close()
    
    # Plot explained variance (1-index x-axis for readability)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Explained Variance - {area_name}')
    plt.savefig(f"plots/pca_variance_{area_name}.png")
    plt.close()

def run_phate(X, y_s, area_name, sentences):
    print(f"Running PHATE on {area_name}...")
    phate_op = phate.PHATE()
    X_phate = phate_op.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.jet(np.linspace(0, 1, len(sentences)))
    for s_idx in range(len(sentences)):
        mask = (y_s == s_idx)
        plt.scatter(X_phate[mask, 0], X_phate[mask, 1], c=colors[s_idx:s_idx+1], label=sentences[s_idx], s=5, alpha=0.5)
    
    plt.legend()
    plt.title(f"PHATE - {area_name}")
    plt.savefig(f"plots/phate_{area_name}.png")
    plt.close()

def run_umap(X, y_s, area_name, sentences):
    print(f"Running UMAP on {area_name}...")
    reducer = umap.UMAP()
    X_umap = reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.jet(np.linspace(0, 1, len(sentences)))
    for s_idx in range(len(sentences)):
        mask = (y_s == s_idx)
        plt.scatter(X_umap[mask, 0], X_umap[mask, 1], c=colors[s_idx:s_idx+1], label=sentences[s_idx], s=5, alpha=0.5)
    
    plt.legend()
    plt.title(f"UMAP - {area_name}")
    plt.savefig(f"plots/umap_{area_name}.png")
    plt.close()

def run_dpca(X_dpca, area_name, sentences):
    # X_dpca input shape: (trials, neurons, time, conditions)
    # dPCA expects (neurons, conditions, time) for labels='st', so move the axes
    # and average over trials before fitting.
    
    # 1. Permute to (neurons, conditions, time, trials)
    # Input is (trials, neurons, time, conditions); permutation order (1, 3, 2, 0)
    X_trial = X_dpca.transpose(1, 3, 2, 0)
    
    # 2. Compute mean over last axis (trials)
    X_mean = np.mean(X_trial, axis=-1) # (neurons, conditions, time)
    # dPCA handles centering internally; avoid double-centering here
    
    print(f"Running dPCA on {area_name}...")
    
    try:
        # Use fixed regularization to avoid CV broadcasting issues with small/odd shapes
        # 'st' tells dPCA to separate variance into stimulus (s), time (t), and their interaction (st)
        # Here 'stimulus' corresponds to which sentence was presented. The interaction 'st' is the time-varying differences between sentences. How the dynamics diver across sentences over time.
        dpca = dPCA.dPCA(labels='st', regularizer=None) 
        dpca.protect = ['t']
        
        Z = dpca.fit_transform(X_mean)
        
        # Visualize top components
        plt.figure(figsize=(12, 4))
        
        # Plot first 3 marginal components
        # Z should be a dictionary with keys 't', 's', 'st'
        keys = list(Z.keys())
        
        for i, key in enumerate(keys[:3]): # Plot first 3 marginalizations found
            data = Z[key]
            # data shape (n_components, n_conditions, n_time)
            
            plt.subplot(1, 3, i+1)
            plt.title("Marginal: " + str(key))
            
            # Plot first component of this marginalization
            for s_idx in range(data.shape[1]):
                plt.plot(data[0, s_idx, :], label=sentences[s_idx] if (i==0 and key != 't') else "")
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"plots/dpca_{area_name}.png")
        plt.close()
        
    except Exception as e:
        print(f"dPCA failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    data, metadata, sentences = load_data()
    
    areas_to_analyze = ['LEX', 'VERB', 'SUBJ']
    
    for area in areas_to_analyze:
        print(f"--- Analyzing {area} ---")
        if area not in metadata: 
            print(f"Skipping {area} (not in metadata)")
            continue
            
        n_neurons = metadata[area]['n']
        X_concat, y_s, y_t, X_dpca, sentences = get_area_matrix(data, area, n_neurons, sentences)
        
        run_pca(X_concat, y_s, y_t, area, sentences)
        
        # PHATE and UMAP
        try:
            run_phate(X_concat, y_s, area, sentences)
        except Exception as e:
            print(f"PHATE failed: {e}")

        try:
            run_umap(X_concat, y_s, area, sentences)
        except Exception as e:
            print(f"UMAP failed: {e}")

        # dPCA
        run_dpca(X_dpca, area, sentences)

if __name__ == "__main__":
    main()
