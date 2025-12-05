import numpy as np
import sys
import os
import random

# Ensure we can import nemo modules
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'nemo'))

import nemo.brain as brain

# Define Areas
ROLE_AGENT = "ROLE_AGENT"
ROLE_PATIENT = "ROLE_PATIENT"
ROLE_ACTION = "ROLE_ACTION"
SEQ = "SEQ"
MOOD = "MOOD"

class RoleSyntaxBrain(brain.Brain):
    def __init__(self, p=0.1, LEX_k=50, beta=0.05): # Lower beta
        brain.Brain.__init__(self, p, seed=42)
        
        # Areas
        self.add_explicit_area(ROLE_AGENT, 2*LEX_k, LEX_k, beta)
        self.add_explicit_area(ROLE_PATIENT, 2*LEX_k, LEX_k, beta)
        self.add_explicit_area(ROLE_ACTION, 2*LEX_k, LEX_k, beta)
        
        # SEQ to learn the order
        self.add_area(SEQ, 5000, LEX_k, beta)
        self.add_explicit_area(MOOD, 100, LEX_k, beta)
        
        self.proj_rounds = 5
        
    def train_sentence(self, sequence_of_roles):
        # sequence_of_roles is list of area names, e.g. [ROLE_AGENT, ROLE_ACTION]
        
        # 1. Activate MOOD -> SEQ
        self.activate(MOOD, 0)
        self.project({}, {MOOD: [SEQ]})
        for _ in range(self.proj_rounds):
            self.project({}, {MOOD: [SEQ], SEQ: [SEQ]})
            
        # 2. Process Sequence
        for role_area in sequence_of_roles:
            # Activate the Role (simulating the grounded word firing)
            self.activate(role_area, 0) # Use index 0 (generic activation)
            
            # Project: SEQ <-> Role
            # The SEQ area updates its state based on current SEQ state + Input Role
            # And reinforces connection SEQ -> Role
            
            self.project({}, {SEQ: [role_area, SEQ], role_area: [SEQ]})
            
            for _ in range(self.proj_rounds):
                self.project({}, {SEQ: [SEQ], role_area: [SEQ]})
                
    def test_generation(self):
        output_sequence = []
        self.disable_plasticity = True
        
        # Clear everything
        for area in self.area_by_name:
            self.area_by_name[area].winners = []
        
        # Start
        self.activate(MOOD, 0)
        self.project({}, {MOOD: [SEQ]})
        # self.project({}, {SEQ: [SEQ]}) # Settle
        
        # Step 1: Predict first role
        self.project({}, {SEQ: [ROLE_AGENT, ROLE_PATIENT, ROLE_ACTION]})
        winner = self.get_winner_role()
        output_sequence.append(winner)
        
        # Step 2
        if winner:
            # Feed back
            self.activate(winner, 0)
            self.project({}, {winner: [SEQ], SEQ: [SEQ]}) # Advance SEQ state
            
            # Predict second
            self.project({}, {SEQ: [ROLE_AGENT, ROLE_PATIENT, ROLE_ACTION]})
            winner = self.get_winner_role()
            output_sequence.append(winner)
            
            # Step 3
            if winner:
                self.activate(winner, 0)
                self.project({}, {winner: [SEQ], SEQ: [SEQ]})
                self.project({}, {SEQ: [ROLE_AGENT, ROLE_PATIENT, ROLE_ACTION]})
                winner = self.get_winner_role()
                output_sequence.append(winner)

        self.disable_plasticity = False
        return output_sequence

    def get_winner_role(self):
        # Find which role area has the MOST active neurons (simplest readout)
        max_winners = 0
        best_area = None
        for area in [ROLE_AGENT, ROLE_PATIENT, ROLE_ACTION]:
            n_winners = len(self.area_by_name[area].winners)
            if n_winners > max_winners:
                max_winners = n_winners
                best_area = area
        
        # Threshold to avoid noise
        if max_winners > 1: # Lower threshold
            return best_area
        return None

def run_experiment(language_type="SVO", num_trials=20):
    print(f"\n--- Running Experiment: {language_type} ---")
    
    # Define Language Rules
    if language_type == "SVO":
        # Consistent start: Agent
        transitive_seq = [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]
        intransitive_seq = [ROLE_AGENT, ROLE_ACTION]
    elif language_type == "OVS":
        # Inconsistent start: Patient vs Agent (assuming Intransitive is SV)
        transitive_seq = [ROLE_PATIENT, ROLE_ACTION, ROLE_AGENT]
        intransitive_seq = [ROLE_AGENT, ROLE_ACTION]
        
    success_count = 0
    
    for trial in range(num_trials):
        brain = RoleSyntaxBrain(p=0.2, LEX_k=50, beta=0.01) 
        
        # Training Schedule
        # Mix of Transitive and Intransitive
        n_epochs = 100
        for _ in range(n_epochs):
            # Train Transitive
            brain.train_sentence(transitive_seq)
            # Train Intransitive
            brain.train_sentence(intransitive_seq)
            
        # Test
        # We test if it can generate the Transitive sentence correctly
        # (Since Transitive is the complex one that requires unique path)
        generated = brain.test_generation()
        
        # Check match
        # We need to filter None
        generated = [g for g in generated if g is not None]
        
        if generated == transitive_seq:
            success_count += 1
        else:
            print(f"Trial {trial} Fail: Expected {transitive_seq}, Got {generated}")
            
    success_rate = (success_count / num_trials) * 100
    print(f"Success Rate for {language_type}: {success_rate}%")
    return success_rate

if __name__ == "__main__":
    run_experiment("SVO")
    run_experiment("OVS")
