import json
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'nemo'))

import nemo.learner as learner

class Recorder:
    def __init__(self):
        self.history = {} # area_name -> list of winners
        self.area_metadata = {} 

    def record(self, b):
        if not self.area_metadata:
            for name, area in b.area_by_name.items():
                self.area_metadata[name] = {'n': area.n, 'k': area.k, 'explicit': area.explicit}
                self.history[name] = []
        
        for name, area in b.area_by_name.items():
            self.history[name].append(list(area.winners))

def run_learner_experiment(config_path):
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    num_trials = config.get("num_trials", 1)
    sentences_to_test = config.get("sentences", [])
    
    final_data = {}
    final_data['metadata'] = None
    # We will record sentences as strings "WORD1 WORD2"
    final_data['sentences'] = [" ".join(s) for s in sentences_to_test]
    
    global_seed_offset = 100

    for s_idx, sentence in enumerate(sentences_to_test):
        for t_idx in range(num_trials):
            print(f"Running: Sentence {s_idx} ({sentence}), Trial {t_idx}...")
            
            seed = 42 + global_seed_offset
            global_seed_offset += 1
            
            brain = learner.LearnBrain(
                config.get("p", 0.05),
                LEX_k=config.get("LEX_k", 50),
                LEX_n=config.get("LEX_n", 10000),
                num_nouns=config.get("num_nouns", 2),
                num_verbs=config.get("num_verbs", 2),
                beta=config.get("beta", 0.06),
                seed=seed
            )
            
            # Train
            training_rounds = config.get("training_rounds", 30)
            brain.train_simple(training_rounds)
            
            # Record processing of the specific sentence
            recorder = Recorder()
            brain.parse_sentence(sentence, step_callback=recorder.record)
            
            if final_data['metadata'] is None:
                final_data['metadata'] = recorder.area_metadata
            
            for area, history in recorder.history.items():
                key = f"{area}_s{s_idx}_t{t_idx}"
                final_data[key] = np.array(history, dtype=object)

    output_file = config.get("output_file", "learner_data.npz")
    np.savez(output_file, **final_data)
    print(f"Experiment finished. Data saved to {output_file}")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "learner_config.json"
    run_learner_experiment(config_path)
