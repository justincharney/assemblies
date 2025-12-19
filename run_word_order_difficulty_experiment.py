"""
Word Order Learning Difficulty Experiment

Based on Mitropolsky & Papadimitriou 2024: "Simulated Language Acquisition
in a Biologically Realistic Model of the Brain"

This experiment investigates why certain word orders (OVS) are harder to learn
than others (SVO) by measuring:
1. Success rate of generating withheld sentences
2. Manifold geometry/topology of neural representations

Key hypothesis: OVS creates inconsistency between transitive (Patient first)
and intransitive (Agent first) sentences, causing interference.
"""

import numpy as np
import random
import json
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'nemo'))

import nemo.brain as brain

# Area name constants (matching paper Figure 4)
PHON = "PHON"               # Phonological input (explicit)
LEX_NOUN = "LEX_NOUN"       # Lexical area for nouns (non-explicit)
LEX_VERB = "LEX_VERB"       # Lexical area for verbs (non-explicit)
VISUAL = "VISUAL"           # Visual context - nouns (explicit)
MOTOR = "MOTOR"             # Motor context - verbs (explicit)

# Thematic role areas (in mutual inhibition during generation)
ROLE_AGENT = "ROLE_AGENT"
ROLE_ACTION = "ROLE_ACTION"
ROLE_PATIENT = "ROLE_PATIENT"
ROLE_SCENE = "ROLE_SCENE"

# Syntactic areas
SUBJ = "SUBJ"
VERB_SYNTAX = "VERB_SYNTAX"
OBJ = "OBJ"

# Control area
MOOD = "MOOD"


class WordOrderRecorder:
    """Records neural activity for manifold analysis."""

    def __init__(self, areas_to_record=None):
        self.areas_to_record = areas_to_record
        self.recordings = {}  # {sentence_label: {area: [winner_lists]}}
        self.area_metadata = {}
        self.current_sentence = None
        self.current_trial = 0

    def start_sentence(self, sentence_label, trial_idx):
        """Start recording a new sentence."""
        self.current_sentence = sentence_label
        self.current_trial = trial_idx
        key = f"{sentence_label}_t{trial_idx}"
        if key not in self.recordings:
            self.recordings[key] = {}

    def record(self, brain_instance):
        """Record current state of all areas."""
        if self.current_sentence is None:
            return

        key = f"{self.current_sentence}_t{self.current_trial}"

        # Initialize metadata on first call
        if not self.area_metadata:
            for name, area in brain_instance.area_by_name.items():
                if self.areas_to_record is None or name in self.areas_to_record:
                    self.area_metadata[name] = {
                        'n': area.n,
                        'k': area.k,
                        'explicit': area.explicit
                    }

        # Record winners for each area
        for name, area in brain_instance.area_by_name.items():
            if self.areas_to_record is None or name in self.areas_to_record:
                if name not in self.recordings[key]:
                    self.recordings[key][name] = []
                self.recordings[key][name].append(list(area.winners))

    def end_sentence(self):
        """End recording current sentence."""
        self.current_sentence = None

    def get_data_for_analysis(self, sentences):
        """
        Convert recordings to format compatible with analyze_manifolds.py

        Returns dict with:
        - metadata: area properties
        - sentences: list of sentence labels
        - {area_name}_s{sent_idx}_t{trial_idx}: winner arrays
        """
        data = {'metadata': self.area_metadata, 'sentences': sentences}

        for sent_idx, sent_label in enumerate(sentences):
            for trial_idx in range(10):  # Check for trials
                key = f"{sent_label}_t{trial_idx}"
                if key in self.recordings:
                    for area_name, winners_list in self.recordings[key].items():
                        data_key = f"{area_name}_s{sent_idx}_t{trial_idx}"
                        data[data_key] = np.array(winners_list, dtype=object)

        return data


class WordOrderBrain(brain.Brain):
    """
    Brain architecture for word order learning experiment.
    Implements Figure 4 from Mitropolsky & Papadimitriou 2024.
    """

    def __init__(self,
                 p=0.05,
                 n_neurons=10000,
                 k=50,
                 beta=0.05,
                 num_nouns=2,
                 num_trans_verbs=2,
                 num_intrans_verbs=2,
                 word_order="SVO",
                 proj_rounds=5,
                 seed=42):
        """
        Initialize word order learning brain.

        Args:
            p: Connection probability
            n_neurons: Neurons per non-explicit area
            k: Winners per area (assembly size)
            beta: Plasticity parameter
            num_nouns: Number of nouns in lexicon
            num_trans_verbs: Number of transitive verbs
            num_intrans_verbs: Number of intransitive verbs
            word_order: "SVO" or "OVS"
            proj_rounds: Projection rounds per step
            seed: Random seed
        """
        brain.Brain.__init__(self, p, seed=seed)

        self.word_order = word_order
        self.num_nouns = num_nouns
        self.num_trans_verbs = num_trans_verbs
        self.num_intrans_verbs = num_intrans_verbs
        self.num_verbs = num_trans_verbs + num_intrans_verbs
        self.lexicon_size = num_nouns + self.num_verbs
        self.proj_rounds = proj_rounds
        self.k = k

        # Word indices
        # Nouns: 0 to num_nouns-1
        # Trans verbs: num_nouns to num_nouns + num_trans_verbs - 1
        # Intrans verbs: num_nouns + num_trans_verbs to lexicon_size - 1
        self.noun_indices = list(range(num_nouns))
        self.trans_verb_indices = list(range(num_nouns, num_nouns + num_trans_verbs))
        self.intrans_verb_indices = list(range(num_nouns + num_trans_verbs, self.lexicon_size))

        self._create_areas(n_neurons, k, beta)

    def _create_areas(self, n, k, beta):
        """Create all brain areas per Figure 4."""

        # Explicit areas (pre-initialized assemblies)
        self.add_explicit_area(PHON, self.lexicon_size * k, k, beta)
        self.add_explicit_area(VISUAL, self.num_nouns * k, k, beta)
        self.add_explicit_area(MOTOR, self.num_verbs * k, k, beta)
        self.add_explicit_area(MOOD, k, k, beta)

        # Non-explicit areas (learn representations)
        self.add_area(LEX_NOUN, n, k, beta)
        self.add_area(LEX_VERB, n, k, beta)

        # Role areas
        self.add_area(ROLE_AGENT, n, k, beta)
        self.add_area(ROLE_ACTION, n, k, beta)
        self.add_area(ROLE_PATIENT, n, k, beta)
        self.add_area(ROLE_SCENE, n, k, beta)

        # Syntactic areas
        self.add_area(SUBJ, n, k, beta)
        self.add_area(VERB_SYNTAX, n, k, beta)
        self.add_area(OBJ, n, k, beta)

    def clear_all_winners(self):
        """Clear winners in all areas."""
        for area in self.area_by_name.values():
            area.winners = []

    def get_input_to_area(self, to_area):
        """Get total synaptic input to an area from all connected areas."""
        total = 0.0
        to_winners = self.area_by_name[to_area].winners
        if not to_winners:
            return 0.0

        for from_area in self.connectomes:
            if to_area in self.connectomes[from_area]:
                from_winners = self.area_by_name[from_area].winners
                if from_winners:
                    connectome = self.connectomes[from_area][to_area]
                    for w in from_winners:
                        for u in to_winners:
                            if w < connectome.shape[0] and u < connectome.shape[1]:
                                total += connectome[w, u]
        return total

    def setup_scene(self, agent_idx, action_idx, patient_idx=None):
        """
        Set up scene in ROLE areas.

        This simulates the TPJ computing thematic roles from visual/motor input.

        Args:
            agent_idx: Index of agent noun (0 to num_nouns-1)
            action_idx: Index of action verb (num_nouns to lexicon_size-1)
            patient_idx: Index of patient noun (None for intransitive)
        """
        self.clear_all_winners()

        # Activate semantic context and project to role areas
        # Agent: VISUAL[agent] -> LEX_NOUN -> ROLE_AGENT
        self.activate(VISUAL, agent_idx)
        self.project({}, {VISUAL: [LEX_NOUN]})
        for _ in range(self.proj_rounds):
            self.project({}, {VISUAL: [LEX_NOUN], LEX_NOUN: [LEX_NOUN, ROLE_AGENT]})

        # Action: MOTOR[action] -> LEX_VERB -> ROLE_ACTION
        motor_idx = action_idx - self.num_nouns
        self.activate(MOTOR, motor_idx)
        self.project({}, {MOTOR: [LEX_VERB]})
        for _ in range(self.proj_rounds):
            self.project({}, {MOTOR: [LEX_VERB], LEX_VERB: [LEX_VERB, ROLE_ACTION]})

        # Patient (if transitive)
        if patient_idx is not None:
            self.activate(VISUAL, patient_idx)
            self.project({}, {VISUAL: [LEX_NOUN]})
            for _ in range(self.proj_rounds):
                self.project({}, {VISUAL: [LEX_NOUN], LEX_NOUN: [LEX_NOUN, ROLE_PATIENT]})

        # Connect all active roles to ROLE_SCENE
        proj_map = {ROLE_AGENT: [ROLE_SCENE], ROLE_ACTION: [ROLE_SCENE]}
        if patient_idx is not None:
            proj_map[ROLE_PATIENT] = [ROLE_SCENE]
        for _ in range(self.proj_rounds):
            self.project({}, proj_map)

    def train_sentence(self, agent_idx, action_idx, patient_idx=None, record_callback=None):
        """
        Train on a single grounded sentence.

        Args:
            agent_idx: PHON index of agent noun
            action_idx: PHON index of action verb
            patient_idx: PHON index of patient noun (None for intransitive)
            record_callback: Optional callback after each projection
        """
        is_transitive = patient_idx is not None

        # Setup scene in ROLE areas
        self.setup_scene(agent_idx, action_idx, patient_idx)

        # Determine word sequence based on word order
        if is_transitive:
            if self.word_order == "SVO":
                word_sequence = [
                    (agent_idx, ROLE_AGENT, SUBJ),
                    (action_idx, ROLE_ACTION, VERB_SYNTAX),
                    (patient_idx, ROLE_PATIENT, OBJ)
                ]
            else:  # OVS
                word_sequence = [
                    (patient_idx, ROLE_PATIENT, OBJ),
                    (action_idx, ROLE_ACTION, VERB_SYNTAX),
                    (agent_idx, ROLE_AGENT, SUBJ)
                ]
        else:  # Intransitive - always SV (Agent-Action)
            word_sequence = [
                (agent_idx, ROLE_AGENT, SUBJ),
                (action_idx, ROLE_ACTION, VERB_SYNTAX)
            ]

        # Activate MOOD to start
        self.activate(MOOD, 0)
        self.project({}, {MOOD: [SUBJ, VERB_SYNTAX, OBJ]})
        if record_callback:
            record_callback(self)

        # Train MOOD -> first role connection
        # This is how the brain learns which role comes first for each word order
        first_role = word_sequence[0][1]  # role_area of first word
        for _ in range(self.proj_rounds):
            self.project({}, {MOOD: [first_role]})
        if record_callback:
            record_callback(self)

        # Process each word in sequence
        prev_syntax_area = None
        for word_idx, role_area, syntax_area in word_sequence:
            # Fire PHON[word]
            self.activate(PHON, word_idx)

            # Determine lexical area
            lex_area = LEX_NOUN if word_idx < self.num_nouns else LEX_VERB

            # Build projection map
            proj_map = {
                PHON: [lex_area],
                role_area: [syntax_area],
                MOOD: [syntax_area]
            }

            # Learning: previous syntax area -> current role area
            # This encodes the word order
            if prev_syntax_area is not None:
                proj_map[prev_syntax_area] = [role_area, syntax_area]

            # Project multiple rounds for plasticity
            for _ in range(self.proj_rounds):
                self.project({}, proj_map)
                if record_callback:
                    record_callback(self)

            prev_syntax_area = syntax_area

    def get_winning_role(self):
        """
        Find which role area has highest activation.
        Implements mutual inhibition - only one role wins.
        """
        role_areas = [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]
        max_winners = -1
        winner = None

        for role in role_areas:
            n_winners = len(self.area_by_name[role].winners)
            if n_winners > max_winners:
                max_winners = n_winners
                winner = role

        # Need minimum activation to count
        if max_winners < self.k // 2:
            return None
        return winner

    def get_total_connection_strength(self, from_area, to_area):
        """
        Calculate total synaptic connection strength from one area to another.
        Uses the connectome weights from winners in from_area to all neurons in to_area.
        """
        from_winners = self.area_by_name[from_area].winners
        if not from_winners:
            return 0.0

        if from_area not in self.connectomes:
            return 0.0
        if to_area not in self.connectomes[from_area]:
            return 0.0

        connectome = self.connectomes[from_area][to_area]
        total = 0.0
        for w in from_winners:
            if w < connectome.shape[0]:
                # Sum all connection weights from this winner neuron
                total += np.sum(connectome[w, :])
        return total

    def generate_sentence(self, agent_idx, action_idx, patient_idx=None):
        """
        Generate a sentence from a scene.

        Mechanism: MOOD -> ROLE connections learned during training
        determine which role fires first. Stronger connections (from more
        training) lead to higher total synaptic weight.

        Returns:
            List of role area names in generated order
        """
        is_transitive = patient_idx is not None
        expected_length = 3 if is_transitive else 2

        # Disable plasticity during generation
        self.disable_plasticity = True

        # Setup scene - this creates assemblies representing the meaning
        self.setup_scene(agent_idx, action_idx, patient_idx)

        # Save role assemblies for later use
        saved_assemblies = {}
        for role in [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]:
            saved_assemblies[role] = list(self.area_by_name[role].winners)

        generated_roles = []
        role_to_syntax = {
            ROLE_AGENT: SUBJ,
            ROLE_ACTION: VERB_SYNTAX,
            ROLE_PATIENT: OBJ
        }

        used_roles = set()
        available_roles = [ROLE_AGENT, ROLE_ACTION]
        if is_transitive:
            available_roles.append(ROLE_PATIENT)

        # Activate MOOD to start generation
        self.activate(MOOD, 0)

        for i in range(expected_length):
            proj_targets = [r for r in available_roles if r not in used_roles]
            if not proj_targets:
                break

            if i == 0:
                # First word: Use actual connection strength from MOOD to each role
                # This reflects the Hebbian learning from training
                best_role = None
                best_strength = -1

                for role in proj_targets:
                    strength = self.get_total_connection_strength(MOOD, role)
                    if strength > best_strength:
                        best_strength = strength
                        best_role = role

            else:
                # Next: Previous syntax area determines next role
                prev_syntax = role_to_syntax[generated_roles[-1]]

                # Use connection strength from previous syntax to remaining roles
                best_role = None
                best_strength = -1

                for role in proj_targets:
                    strength = self.get_total_connection_strength(prev_syntax, role)
                    if strength > best_strength:
                        best_strength = strength
                        best_role = role

            if best_role is None:
                # Fallback: pick first available role
                for role in proj_targets:
                    best_role = role
                    break

            if best_role is None:
                break

            generated_roles.append(best_role)
            used_roles.add(best_role)

            # Restore the semantic assembly for the winning role
            self.area_by_name[best_role].winners = saved_assemblies[best_role]

            # Project winner role to its syntax area to strengthen connections
            syntax_area = role_to_syntax[best_role]
            if self.area_by_name[best_role].winners:
                for _ in range(self.proj_rounds):
                    self.project({}, {best_role: [syntax_area]})

        self.disable_plasticity = False
        return generated_roles

    def test_generation(self, agent_idx, action_idx, patient_idx=None):
        """
        Test if generated order matches expected order.

        Returns:
            (success: bool, generated: list, expected: list)
        """
        generated = self.generate_sentence(agent_idx, action_idx, patient_idx)

        is_transitive = patient_idx is not None
        if is_transitive:
            if self.word_order == "SVO":
                expected = [ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT]
            else:  # OVS
                expected = [ROLE_PATIENT, ROLE_ACTION, ROLE_AGENT]
        else:  # Intransitive - always SV
            expected = [ROLE_AGENT, ROLE_ACTION]

        success = generated == expected
        return success, generated, expected


def run_training(brain_instance, num_sentences, trans_ratio,
                 withheld_trans, withheld_intrans, recorder=None):
    """
    Main training loop.

    Args:
        brain_instance: WordOrderBrain instance
        num_sentences: Total number of training sentences
        trans_ratio: Ratio of transitive sentences (0.0 to 1.0)
        withheld_trans: (agent, verb, patient) tuple to withhold
        withheld_intrans: (agent, verb) tuple to withhold
        recorder: Optional WordOrderRecorder
    """
    # Generate all possible sentences (excluding withheld)
    all_trans = []
    for agent in brain_instance.noun_indices:
        for verb in brain_instance.trans_verb_indices:
            for patient in brain_instance.noun_indices:
                if agent != patient:  # Agent != Patient
                    sent = (agent, verb, patient)
                    if sent != withheld_trans:
                        all_trans.append(sent)

    all_intrans = []
    for agent in brain_instance.noun_indices:
        for verb in brain_instance.intrans_verb_indices:
            sent = (agent, verb)
            if sent != withheld_intrans:
                all_intrans.append(sent)

    # Calculate how many of each type
    num_trans = int(num_sentences * trans_ratio)
    num_intrans = num_sentences - num_trans

    # Build training set by sampling with replacement if needed
    training_sentences = []

    for _ in range(num_trans):
        sent = random.choice(all_trans)
        training_sentences.append(('trans', sent))

    for _ in range(num_intrans):
        sent = random.choice(all_intrans)
        training_sentences.append(('intrans', sent))

    # Shuffle
    random.shuffle(training_sentences)

    # Train
    for idx, (sent_type, sent) in enumerate(training_sentences):
        if recorder:
            label = f"train_{sent_type}_{idx}"
            recorder.start_sentence(label, 0)
            callback = recorder.record
        else:
            callback = None

        if sent_type == 'trans':
            brain_instance.train_sentence(sent[0], sent[1], sent[2], callback)
        else:
            brain_instance.train_sentence(sent[0], sent[1], None, callback)

        if recorder:
            recorder.end_sentence()


def run_single_experiment(word_order, config, trial_idx, record=False):
    """
    Run a single experiment for one word order.

    Returns:
        dict with success metrics and optional recordings
    """
    brain_instance = WordOrderBrain(
        p=config['p'],
        n_neurons=config['n'],
        k=config['k'],
        beta=config['beta'],
        num_nouns=config['num_nouns'],
        num_trans_verbs=config['num_trans_verbs'],
        num_intrans_verbs=config['num_intrans_verbs'],
        word_order=word_order,
        proj_rounds=config['proj_rounds'],
        seed=config['seed_base'] + trial_idx
    )

    random.seed(config['seed_base'] + trial_idx)
    np.random.seed(config['seed_base'] + trial_idx)

    # Select withheld sentences randomly
    withheld_trans = (
        random.choice(brain_instance.noun_indices),
        random.choice(brain_instance.trans_verb_indices),
        random.choice(brain_instance.noun_indices)
    )
    # Ensure agent != patient
    while withheld_trans[0] == withheld_trans[2]:
        withheld_trans = (
            withheld_trans[0],
            withheld_trans[1],
            random.choice(brain_instance.noun_indices)
        )

    withheld_intrans = (
        random.choice(brain_instance.noun_indices),
        random.choice(brain_instance.intrans_verb_indices)
    )

    # Setup recorder if needed
    recorder = None
    if record:
        recorder = WordOrderRecorder(areas_to_record=[
            ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT,
            SUBJ, VERB_SYNTAX, OBJ
        ])

    # Train
    run_training(
        brain_instance,
        num_sentences=config['num_sentences'],
        trans_ratio=config['trans_ratio'],
        withheld_trans=withheld_trans,
        withheld_intrans=withheld_intrans,
        recorder=recorder
    )

    # Test on withheld sentences
    trans_success, trans_gen, trans_exp = brain_instance.test_generation(*withheld_trans)
    intrans_success, intrans_gen, intrans_exp = brain_instance.test_generation(*withheld_intrans)

    result = {
        'trans_success': trans_success,
        'intrans_success': intrans_success,
        'overall_success': trans_success and intrans_success,
        'trans_generated': [str(r) for r in trans_gen],
        'trans_expected': [str(r) for r in trans_exp],
        'intrans_generated': [str(r) for r in intrans_gen],
        'intrans_expected': [str(r) for r in intrans_exp],
        'withheld_trans': withheld_trans,
        'withheld_intrans': withheld_intrans
    }

    if recorder:
        result['recorder'] = recorder

    return result


def run_full_experiment(output_dir="word_order_experiment_results"):
    """Main experiment runner."""

    os.makedirs(output_dir, exist_ok=True)

    # Parameters from paper
    base_config = {
        'p': 0.05,
        'n': 10000,
        'k': 50,
        'beta': 0.05,
        'num_nouns': 2,
        'num_trans_verbs': 2,
        'num_intrans_verbs': 2,
        'num_sentences': 50,  # Paper needed ~44 for OVS
        'trans_ratio': 0.7,   # 70% transitive
        'proj_rounds': 5,
        'seed_base': 42
    }

    num_trials = 20  # Per paper: "averaging over 20 runs per order"

    results = {'SVO': [], 'OVS': []}

    print("=" * 60)
    print("Word Order Learning Difficulty Experiment")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Training sentences: {base_config['num_sentences']}")
    print(f"  Transitive ratio: {base_config['trans_ratio'] * 100:.0f}%")
    print(f"  Trials per order: {num_trials}")
    print("=" * 60)

    for word_order in ['SVO', 'OVS']:
        print(f"\n=== Running {word_order} experiments ===")

        for trial in range(num_trials):
            # Record on first trial only for manifold analysis
            record = (trial == 0)
            result = run_single_experiment(word_order, base_config, trial, record=record)
            results[word_order].append(result)

            status = "SUCCESS" if result['overall_success'] else "FAIL"
            print(f"  Trial {trial + 1:2d}/{num_trials}: {status}")
            if not result['overall_success']:
                print(f"    Trans: expected {result['trans_expected']}, got {result['trans_generated']}")
                print(f"    Intrans: expected {result['intrans_expected']}, got {result['intrans_generated']}")

        successes = sum(1 for r in results[word_order] if r['overall_success'])
        trans_successes = sum(1 for r in results[word_order] if r['trans_success'])
        intrans_successes = sum(1 for r in results[word_order] if r['intrans_success'])

        print(f"\n{word_order} Results:")
        print(f"  Overall Success Rate: {successes}/{num_trials} = {100*successes/num_trials:.1f}%")
        print(f"  Transitive Success: {trans_successes}/{num_trials} = {100*trans_successes/num_trials:.1f}%")
        print(f"  Intransitive Success: {intrans_successes}/{num_trials} = {100*intrans_successes/num_trials:.1f}%")

    # Save results
    summary = {
        'config': base_config,
        'num_trials': num_trials,
        'results': {
            'SVO': {
                'overall_success_rate': sum(1 for r in results['SVO'] if r['overall_success']) / num_trials,
                'trans_success_rate': sum(1 for r in results['SVO'] if r['trans_success']) / num_trials,
                'intrans_success_rate': sum(1 for r in results['SVO'] if r['intrans_success']) / num_trials,
                'trials': [{k: v for k, v in r.items() if k != 'recorder'} for r in results['SVO']]
            },
            'OVS': {
                'overall_success_rate': sum(1 for r in results['OVS'] if r['overall_success']) / num_trials,
                'trans_success_rate': sum(1 for r in results['OVS'] if r['trans_success']) / num_trials,
                'intrans_success_rate': sum(1 for r in results['OVS'] if r['intrans_success']) / num_trials,
                'trials': [{k: v for k, v in r.items() if k != 'recorder'} for r in results['OVS']]
            }
        }
    }

    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save recordings for manifold analysis (first trial of each)
    for word_order in ['SVO', 'OVS']:
        if 'recorder' in results[word_order][0]:
            recorder = results[word_order][0]['recorder']
            sentences = list(set(key.rsplit('_t', 1)[0] for key in recorder.recordings.keys()))
            data = recorder.get_data_for_analysis(sentences)

            npz_path = os.path.join(output_dir, f"{word_order.lower()}_data.npz")
            np.savez(npz_path, **data)
            print(f"Neural recordings saved to {npz_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"SVO Success Rate: {summary['results']['SVO']['overall_success_rate']*100:.1f}%")
    print(f"OVS Success Rate: {summary['results']['OVS']['overall_success_rate']*100:.1f}%")

    return results, summary


if __name__ == "__main__":
    results, summary = run_full_experiment()
