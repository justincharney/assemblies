# Word Order Learning Difficulty Experiment

This experiment investigates why certain word orders (OVS) are harder to learn than others (SVO) using NEMO, based on Mitropolsky & Papadimitriou 2024.

## Overview

### The Hypothesis

**SVO word order** (Subject-Verb-Object, like English):
- Transitive: "Dog chases cat": Agent-Action-Patient
- Intransitive: "Dog runs": Agent-Action
- **Consistent**: Both start with Agent

**OVS word order** (Object-Verb-Subject, like some ergative languages):
- Transitive: "Cat chases dog": Patient-Action-Agent
- Intransitive: "Dog runs": Agent-Action
- **Inconsistent**: Transitive starts with Patient, intransitive starts with Agent

This inconsistency creates **interference** in neural circuits, making OVS harder to learn.

## Files

### Main Experiment

| File | Description |
|------|-------------|
| `run_word_order_difficulty_experiment.py` | Main experiment runner with `WordOrderBrain` class |
| `word_order_analysis.py` | Manifold analysis comparing SVO vs OVS neural representations |

### Dependencies (existing files)

| File | Description |
|------|-------------|
| `nemo/brain.py` | Core NEMO Brain class with areas, projections, Hebbian learning |
| `analyze_manifolds.py` | Geometric and topological analysis functions |
| `visualize_interactive.py` | Data loading utilities |

## Running the Experiment

### Basic Usage

```bash
# Run the full experiment (20 trials per word order)
python run_word_order_difficulty_experiment.py

# Run manifold analysis on results
python word_order_analysis.py --data-dir word_order_experiment_results
```

### Parameters

Edit `run_word_order_difficulty_experiment.py` to change these parameters in the `base_config` dict:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_sentences` | 50 | Training sentences per trial |
| `trans_ratio` | 0.7 | Fraction of transitive sentences (0.0-1.0) |
| `num_nouns` | 2 | Number of nouns in vocabulary |
| `num_trans_verbs` | 2 | Number of transitive verbs |
| `num_intrans_verbs` | 2 | Number of intransitive verbs |
| `n` | 10000 | Neurons per brain area |
| `k` | 50 | Assembly size (winners per area) |
| `beta` | 0.05 | Hebbian plasticity parameter |
| `p` | 0.05 | Connection probability |
| `proj_rounds` | 5 | Projection iterations per step |
| `seed_base` | 42 | Random seed for reproducibility |

The number of trials is set by `num_trials = 20`.

### Key Parameter Effects

- **trans_ratio < 0.5**: OVS cannot be learned (paper finding)
- **trans_ratio = 0.7**: OVS learnable with ~44 sentences, SVO with ~24

## Output Files

All outputs are saved to `word_order_experiment_results/`:

| File | Description |
|------|-------------|
| `experiment_results.json` | Success rates and trial-by-trial results |
| `svo_data.npz` | Neural recordings for SVO (first trial) |
| `ovs_data.npz` | Neural recordings for OVS (first trial) |
| `manifold_comparison.csv` | Geometric/topological metrics per area |
| `word_order_comparison.png` | Bar chart comparing SVO vs OVS metrics |
| `*_topology.png` | Persistence diagrams per area |

### experiment_results.json Structure

```json
{
  "config": { ... },
  "num_trials": 20,
  "results": {
    "SVO": {
      "overall_success_rate": 1.0,
      "trans_success_rate": 1.0,
      "intrans_success_rate": 1.0,
      "trials": [ ... ]
    },
    "OVS": { ... }
  }
}
```

## Manifold Analysis Metrics

The `word_order_analysis.py` script computes these metrics for each brain area:

### Geometric Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Effective Dimensionality (PR)** | Participation ratio of PCA eigenvalues | Higher = more distributed representation |
| **Dim (95% Var)** | PCA dimensions for 95% variance | Intrinsic dimensionality estimate |
| **Linearity (Geo/Euc Corr)** | Correlation of geodesic vs Euclidean distances | Higher = more linear manifold, Lower = more "tangled" |
| **Avg Local Curvature** | Residual variance from local PCA | Higher = more curved manifold |

### Topological Metrics (Persistent Homology)

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **H0 Feature Count** | Number of connected components | More = fragmented representation |
| **H0 Max/Avg Lifetime** | Persistence of components | Longer = more stable clusters |
| **H1 Feature Count** | Number of loops/holes | Topological complexity |
| **H2 Feature Count** | Number of voids | Higher-order structure |

### Brain Areas Analyzed

| Area | Type | Role |
|------|------|------|
| `ROLE_AGENT` | Thematic | Agent/Actor representation |
| `ROLE_ACTION` | Thematic | Verb/Action representation |
| `ROLE_PATIENT` | Thematic | Patient/Object representation |
| `SUBJ` | Syntactic | Subject position |
| `VERB_SYNTAX` | Syntactic | Verb position |
| `OBJ` | Syntactic | Object position |

## Key Findings

### Behavioral Results

| Condition | SVO | OVS |
|-----------|-----|-----|
| 70% trans, 50 sentences | 100% | 100% |
| 70% trans, 25 sentences | 100% | 100% |
| 50% trans, 25 sentences | 100% | **0%** |

OVS fails at 50% transitive ratio because the conflicting first-role signals (Patient vs Agent) cancel out.

### Manifold Findings

**Original hypothesis**: Failed learning = collapsed manifolds (lower dimensionality, fewer components)

**Actual finding**: Failed OVS shows:
- **Higher** dimensionality (not collapsed)
- **More** H0 components (fragmented, not merged)
- **Lower** linearity (confirmed - more tangled)

**Interpretation**: Interference doesn't collapse representations but creates **confused/disorganized** patterns. The brain uses MORE resources trying to encode conflicting signals.

## Architecture

The `WordOrderBrain` implements Figure 4 from the paper:

### Generation Flow
1. Scene setup (activate roles from semantics)
2. MOOD fires
3. Compare MOOD-ROLE connection strengths
4. Winner role determines first word
5. Continue with SYNTAX-ROLE for subsequent words

## Customization

### Adding More Areas

In `WordOrderBrain._create_areas()`:
```python
self.add_area("NEW_AREA", n, k, beta)
```

### Changing Vocabulary Size

Adjust `num_nouns`, `num_trans_verbs`, `num_intrans_verbs` in config.

### Recording Different Areas

In `run_single_experiment()`, modify the `areas_to_record` list:
```python
recorder = WordOrderRecorder(areas_to_record=[
    ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT,
    SUBJ, VERB_SYNTAX, OBJ,
    # Add more areas here
])
```