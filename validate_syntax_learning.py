import numpy as np
import sys
import os
import json

# Ensure we can import nemo modules
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'nemo'))

import nemo.learner as learner

def validate_syntax_learning(order="NV", num_trials=20):
    print(f"Validating Syntax Learning for Order: {order} ({num_trials} trials)")
    success_count = 0
    
    # Constants from learner.py
    NOUN_CORE = 0
    VERB_CORE = 1
    
    for trial in range(num_trials):
        # Fresh brain each time
        brain = learner.SimpleSyntaxBrain(p=0.1, LEX_k=50, LEX_n=5000)
        brain.pre_train(proj_rounds=20)
        brain.train(order, train_rounds=50) 
        
        # TEST GENERATION
        brain.disable_plasticity = True
        brain.area_by_name[learner.CORE].unfix_assembly()
        
        # Step 1: Start Generation (Trigger first word)
        brain.activate(learner.MOOD, 0)
        brain.project({}, {learner.MOOD: [learner.SEQ]})
        brain.project({}, {learner.SEQ: [learner.CORE]})
        
        first_core = brain.get_explicit_assembly(learner.CORE)
        
        # Step 2: Trigger second word
        brain.project({}, {learner.CORE: [learner.SEQ]})
        brain.project({}, {learner.SEQ: [learner.CORE]})
        
        second_core = brain.get_explicit_assembly(learner.CORE)
        
        # Check correctness
        correct = False
        if order == "NV":
            if first_core == NOUN_CORE and second_core == VERB_CORE:
                correct = True
        elif order == "VN":
            if first_core == VERB_CORE and second_core == NOUN_CORE:
                correct = True
                
        if correct:
            success_count += 1
            # print(f"Trial {trial}: Success")
        else:
            pass
            # print(f"Trial {trial}: Fail (Got {first_core} -> {second_core})")
            
    success_rate = (success_count / num_trials) * 100
    print(f"\nFinal Success Rate for {order}: {success_rate}%")
    return success_rate

if __name__ == "__main__":
    order = sys.argv[1] if len(sys.argv) > 1 else "NV"
    validate_syntax_learning(order)
