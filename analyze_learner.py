import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dPCA import dPCA
import phate
import umap
import os
import sys

# Create output directory
os.makedirs("learner_plots", exist_ok=True)

def load_data(filepath="learner_data.npz"):
    print(f"Loading {filepath}...")
    data = np.load(filepath, allow_pickle=True)
    metadata = data['metadata'].item()
    sentences = data['sentences']
    
    return data, metadata, sentences

def get_area_matrix(data, area_name, n_neurons, sentences, num_trials=2):
    all_winners = set()
    
    for s_idx in range(len(sentences)):
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            if key not in data:
                continue
            history = data[key] 
            for step_winners in history:
                all_winners.update(step_winners)
                
    sorted_winners = sorted(list(all_winners))
    winner_map = {idx: i for i, idx in enumerate(sorted_winners)}
    reduced_n = len(sorted_winners)
    print(f"Area {area_name}: Original N={n_neurons}, Active N={reduced_n}")
    
    expected_length = None
    for s_idx, sentence in enumerate(sentences):
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            if key not in data: continue
            time_len = len(data[key])
            if expected_length is None:
                expected_length = time_len
            elif time_len != expected_length:
                # Should not happen if fixed steps in simulation
                raise RuntimeError(f"Inconsistent time lengths in {area_name}: expected {expected_length}, got {time_len}")
                
    if expected_length is None:
         return None, None, None, None, None
         
    time_steps = expected_length
    X_dpca = np.zeros((num_trials, reduced_n, time_steps, len(sentences)))
    
    X_concat = []
    y_sentence = []
    y_time = []
    
    for s_idx, sentence in enumerate(sentences):
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            history = data[key]
            
            trial_mat = np.zeros((len(history), reduced_n))
            for t, winners in enumerate(history):
                indices = [winner_map[w] for w in winners if w in winner_map]
                trial_mat[t, indices] = 1.0
            
            X_concat.append(trial_mat)
            y_sentence.extend([s_idx] * len(history))
            y_time.extend(list(range(len(history))))
            
            if len(history) == time_steps:
                 X_dpca[t_idx, :, :, s_idx] = trial_mat.T

    X_concat = np.vstack(X_concat)
    y_sentence = np.array(y_sentence)
    y_time = np.array(y_time)
    
    return X_concat, y_sentence, y_time, X_dpca, sentences

def run_pca(X, y_s, area_name, sentences):
    print(f"Running PCA on {area_name}...")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, len(sentences)))
    
    for s_idx in range(len(sentences)):
        mask = (y_s == s_idx)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                   c=colors[s_idx:s_idx+1], label=sentences[s_idx], s=10, alpha=0.6)
        
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.title(f"PCA - {area_name}")
    plt.savefig(f"learner_plots/pca_{area_name}.png")
    plt.close()

def run_phate(X, y_s, area_name, sentences):
    print(f"Running PHATE on {area_name}...")
    try:
        phate_op = phate.PHATE()
        X_phate = phate_op.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.jet(np.linspace(0, 1, len(sentences)))
        for s_idx in range(len(sentences)):
            mask = (y_s == s_idx)
            plt.scatter(X_phate[mask, 0], X_phate[mask, 1], c=colors[s_idx:s_idx+1], label=sentences[s_idx], s=10, alpha=0.6)
        
        plt.legend()
        plt.title(f"PHATE - {area_name}")
        plt.savefig(f"learner_plots/phate_{area_name}.png")
        plt.close()
    except Exception as e:
        print(f"PHATE failed: {e}")

def run_umap(X, y_s, area_name, sentences):
    print(f"Running UMAP on {area_name}...")
    try:
        reducer = umap.UMAP()
        X_umap = reducer.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.jet(np.linspace(0, 1, len(sentences)))
        for s_idx in range(len(sentences)):
            mask = (y_s == s_idx)
            plt.scatter(X_umap[mask, 0], X_umap[mask, 1], c=colors[s_idx:s_idx+1], label=sentences[s_idx], s=10, alpha=0.6)
        
        plt.legend()
        plt.title(f"UMAP - {area_name}")
        plt.savefig(f"learner_plots/umap_{area_name}.png")
        plt.close()
    except Exception as e:
        print(f"UMAP failed: {e}")

def run_dpca(X_dpca, area_name, sentences):
    # X_dpca: (trials, neurons, time, conditions)
    # Permute to (neurons, conditions, time, trials)
    X_trial = X_dpca.transpose(1, 3, 2, 0)
    X_mean = np.mean(X_trial, axis=-1)
    
    
    print(f"Running dPCA on {area_name}...")
    try:
        dpca = dPCA.dPCA(labels='st', regularizer=None)
        dpca.protect = ['t']
        
        Z = dpca.fit_transform(X_mean)
        
        keys = list(Z.keys())
        plt.figure(figsize=(12, 4))
        for i, key in enumerate(keys[:3]):
            data = Z[key]
            plt.subplot(1, 3, i+1)
            plt.title("Marginal: " + str(key))
            for s_idx in range(data.shape[1]):
                plt.plot(data[0, s_idx, :], label=sentences[s_idx] if (i==0 and key!='t') else "")
            if i==0: plt.legend()
            
        plt.tight_layout()
        plt.savefig(f"learner_plots/dpca_{area_name}.png")
        plt.close()
    except Exception as e:
        print(f"dPCA failed: {e}")

def main():
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "learner_data.npz"

    data, metadata, sentences = load_data(data_file)
    
    # Areas in LearnBrain
    # PHON, MOTOR, VISUAL, NOUN, VERB
    areas_to_analyze = ['NOUN', 'VERB']
    
    for area in areas_to_analyze:
        print(f"--- Analyzing {area} ---")
        if area not in metadata: 
            print(f"Skipping {area}")
            continue
            
        n_neurons = metadata[area]['n']
        X_concat, y_s, y_t, X_dpca, sentences = get_area_matrix(data, area, n_neurons, sentences)
        
        if X_concat is None:
            continue

        run_dpca(X_dpca, area, sentences)

if __name__ == "__main__":
    main()
