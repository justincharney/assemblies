import numpy as np
import sys
import os
import json

# Ensure we can import nemo modules
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

def run_syntax_experiment(order="NV", output_file="syntax_data.npz"):
    print(f"Running Syntax Experiment for Order: {order}")
    
    # Parameters matching the paper/default
    # Note: SimpleSyntaxBrain uses NOUN_VERB area
    brain = learner.SimpleSyntaxBrain(p=0.1, LEX_k=50, LEX_n=5000)
    
    print("Pre-training...")
    brain.pre_train(proj_rounds=20)
    
    print(f"Training on {order}...")
    brain.train(order, train_rounds=50)
    
    # Testing & Recording
    # We want to record the "trajectory" of parsing a sentence
    # Sentence is [0, 2] (Cat Jump) for NV or [2, 0] (Jump Cat) for VN
    if order == "NV":
        sentence = [0, 2]
        sentence_str = "CAT JUMP"
    else:
        sentence = [2, 0]
        sentence_str = "JUMP CAT"
        
    recorder = Recorder()
    
    print(f"Recording trajectory for {sentence_str}...")
    
    # Logic from SimpleSyntaxBrain.parse, manually executed to record steps
    mood_state = 0
    brain.activate(learner.MOOD, mood_state)
    brain.project({}, {learner.MOOD: [learner.SEQ]})
    recorder.record(brain) # Record initial state
    
    for _ in range(brain.proj_rounds):
        brain.project({}, {learner.MOOD: [learner.SEQ], learner.SEQ: [learner.SEQ]})
        recorder.record(brain)

    for word in sentence:
        brain.activate(learner.NOUN_VERB, word)
        area_firing_into_core = learner.NOUN_VERB
        brain.project({}, {learner.SEQ: [learner.CORE, learner.SEQ], area_firing_into_core: [learner.CORE]})
        
        brain.project({}, {learner.CORE: [learner.SEQ]})
        recorder.record(brain)
        
        for _ in range(brain.proj_rounds):
            brain.project({}, {learner.CORE: [learner.SEQ], learner.SEQ: [learner.SEQ]})
            recorder.record(brain)

    # Save data
    final_data = {}
    final_data['metadata'] = recorder.area_metadata
    final_data['sentences'] = [sentence_str]
    
    # Flatten history to standard format
    # Using multiple trials to simulate variability? No, deterministic here unless we add noise.
    # Let's run it once.
    for area, history in recorder.history.items():
        key = f"{area}_s0_t0" # Single sentence, trial 0
        final_data[key] = np.array(history, dtype=object)
        
    np.savez(output_file, **final_data)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    order = sys.argv[1] if len(sys.argv) > 1 else "NV"
    output = sys.argv[2] if len(sys.argv) > 2 else f"syntax_{order}.npz"
    run_syntax_experiment(order, output)
