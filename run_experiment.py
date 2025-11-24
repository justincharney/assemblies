import json
import numpy as np
import sys
import os

# Ensure we can import nemo modules and their dependencies
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'nemo'))

import nemo.recursive_parser as rp

class Recorder:
    def __init__(self):
        self.history = {} # area_name -> list of winners (list of lists)
        self.area_metadata = {} # area_name -> {n: int, k: int}

    def record(self, b):
        # Initialize metadata if first run
        if not self.area_metadata:
            for name, area in b.area_by_name.items():
                self.area_metadata[name] = {'n': area.n, 'k': area.k, 'explicit': area.explicit}
                self.history[name] = []
        
        # Record current state
        for name, area in b.area_by_name.items():
            self.history[name].append(list(area.winners))

def run_experiment(config_path):
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    sentences = config.get("sentences", ["cats chase mice"])
    num_trials = config.get("num_trials_per_sentence", 1)
    
    # Store all data
    # Structure: flat dict with keys "area_s{i}_t{j}" -> np array
    final_data = {}
    
    # We'll grab metadata from the first run
    final_data['metadata'] = None
    final_data['sentences'] = sentences

    global_seed_offset = 0

    for s_idx, sentence in enumerate(sentences):
        for t_idx in range(num_trials):
            print(f"Running: Sentence {s_idx} ('{sentence}'), Trial {t_idx}...")
            recorder = Recorder()
            
            # Use a deterministic seed scheme
            seed = 42 + global_seed_offset
            global_seed_offset += 1

            rp.parse(
                sentence=sentence,
                language=config.get("language", "English"),
                p=config.get("p", 0.1),
                LEX_k=config.get("LEX_k", 20),
                project_rounds=config.get("project_rounds", 20),
                verbose=config.get("verbose", False),
                step_callback=recorder.record,
                seed=seed
            )
            
            if final_data['metadata'] is None:
                final_data['metadata'] = recorder.area_metadata
            
            for area, history in recorder.history.items():
                key = f"{area}_s{s_idx}_t{t_idx}"
                # Convert to object array
                final_data[key] = np.array(history, dtype=object)

    # Save results
    output_file = config.get("output_file", "assemblies_data.npz")
    np.savez(output_file, **final_data)
    print(f"Experiment finished. Data saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.json"
    run_experiment(config_path)