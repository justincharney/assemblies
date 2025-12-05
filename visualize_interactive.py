import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
import phate
import os
import sys

# Create output directory
os.makedirs("interactive_plots", exist_ok=True)

def load_data(filepath="learner_data.npz"):
    print(f"Loading {filepath}...")
    data = np.load(filepath, allow_pickle=True)
    metadata = data['metadata'].item()
    sentences = data['sentences']
    return data, metadata, sentences

def get_area_matrix_flat(data, area_name, sentences, num_trials=2):
    """
    Returns a flattened matrix suitable for dimensionality reduction,
    along with metadata columns for plotting.
    """
    all_winners = set()
    for s_idx in range(len(sentences)):
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            if key in data:
                for step_winners in data[key]:
                    all_winners.update(step_winners)
    
    sorted_winners = sorted(list(all_winners))
    winner_map = {idx: i for i, idx in enumerate(sorted_winners)}
    reduced_n = len(sorted_winners)
    print(f"Area {area_name}: Active Neurons = {reduced_n}")
    
    X_list = []
    meta = []

    for s_idx, sentence in enumerate(sentences):
        for t_idx in range(num_trials):
            key = f"{area_name}_s{s_idx}_t{t_idx}"
            if key not in data: continue
            
            history = data[key]
            
            for t, winners in enumerate(history):
                # Create one-hot vector for this time step
                vec = np.zeros(reduced_n)
                indices = [winner_map[w] for w in winners if w in winner_map]
                vec[indices] = 1.0
                
                X_list.append(vec)
                meta.append({
                    "Sentence": sentence,
                    "Trial": str(t_idx),
                    "Time": t,
                    "Label": f"{sentence} (T{t_idx})",
                    "Area": area_name
                })

    X = np.vstack(X_list)
    df = pd.DataFrame(meta)
    return X, df

def plot_interactive_3d(X_reduced, df, method_name, area_name):
    print(f"Generating interactive {method_name} plot for {area_name}...")
    
    # Add coordinates to dataframe
    df['x'] = X_reduced[:, 0]
    df['y'] = X_reduced[:, 1]
    df['z'] = X_reduced[:, 2]

    # Create 3D Line Plot
    # line_group="Label" ensures connecting lines are drawn per trial
    # color="Sentence" ensures different sentences get different colors
    fig = px.line_3d(
        df, 
        x='x', y='y', z='z',
        color='Sentence', 
        line_group='Label',
        hover_data=['Time', 'Sentence'],
        title=f"{area_name} - {method_name} Manifold",
        markers=True # Show points as well as lines
    )
    
    # Update markers to be smaller
    fig.update_traces(marker=dict(size=3), line=dict(width=4))
    
    output_path = f"interactive_plots/{area_name}_{method_name}.html"
    fig.write_html(output_path)
    print(f"Saved to {output_path}")

def run_pca_3d(X, df, area_name):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    plot_interactive_3d(X_pca, df, "PCA", area_name)

def run_phate_3d(X, df, area_name):
    # PHATE naturally works well in 3D for branching
    phate_op = phate.PHATE(n_components=3, knn=5, decay=20)
    try:
        X_phate = phate_op.fit_transform(X)
        plot_interactive_3d(X_phate, df, "PHATE", area_name)
    except Exception as e:
        print(f"PHATE failed for {area_name}: {e}")

def main():
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "learner_data.npz"

    data, metadata, sentences = load_data(data_file)
    
    # Areas to visualize
    # Focusing on the latent areas (NOUN, VERB) and the input (PHON)
    areas = ['PHON', 'NOUN', 'VERB'] 
    
    for area in areas:
        if area not in metadata: continue
        
        X, df = get_area_matrix_flat(data, area, sentences, num_trials=2)
        if X is None or len(X) == 0: continue

        run_pca_3d(X, df, area)
        run_phate_3d(X, df, area)

if __name__ == "__main__":
    main()
