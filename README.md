# Balancing Relevance and Redundancy: An Optuna-Tuned QUBO Formulation for Feature Selection in Recommender Systems

**Team Skywalkers — QuantumCLEF 2026 Task 1B submission**

> Bhavishya Prajapat and Alapan Kuila  
> Indian Institute of Information Technology, Design and Manufacturing, Kurnool

[![CLEF 2026](https://img.shields.io/badge/CLEF-2026-blue)](https://clef2026.clef-initiative.eu/)
[![QuantumCLEF](https://img.shields.io/badge/QuantumCLEF-Task%201B-teal)](https://quantum-clef.github.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

This repository contains the code for our submission to [QuantumCLEF 2026 Task 1B](https://quantum-clef.github.io/) (Feature Selection for Recommender Systems). We reformulate item feature selection as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and automatically calibrate the penalty weights (Lagrange multipliers) using **Optuna Bayesian optimization**, evaluated with both **Simulated Annealing (SA)** and a physical **D-Wave Quantum Processing Unit (QPU)**.

**Key results on the official leaderboard (`100_ICM` dataset):**

| Solver | Best NDCG@10 | Features (k) |
|--------|-------------|--------------|
| Simulated Annealing (SA) | **0.0287** ✅ #1 | 59 |
| Quantum Annealing (QPU) | 0.0211 | 59 |
| Classical baseline (k=100) | 0.0226 | 100 |

---

## Method Summary

```
ICM (100 features)
      │
      ▼
[1] Compute QUBO coefficients
      ├── Relevance R_i  ← Random Forest (Gini importance)
      └── Redundancy S_ij ← Pairwise cosine similarity
      │
      ▼
[2] Optuna (1000 trials, TPE)
      └── Suggests α, β, λ → builds QUBO matrix Q
            │
            ▼
      [3a] Simulated Annealing (dwave-neal, 250 reads)
      [3b] D-Wave QPU (EmbeddingComposite, 1000 reads)  ← requires D-Wave Leap
            │
            ▼
      Feature subset x (|x| = k)
            │
            ▼
[4] Item-KNN (topK=100, cosine, shrink=5)
      └── NDCG@10 → feedback to Optuna
```

---

## Repository Structure

```
code/
├── README.md
├── requirements.txt
├── CITATION.cff
├── feature_selection_qubo.py   # Core QUBO pipeline (SA-compatible, no QPU required)
└── notebooks/
    └── qCLEF_submission.ipynb  # Original submission notebook (requires qCLEF workspace)
```

---

## Requirements

```bash
pip install -r requirements.txt
```

The qCLEF framework (`Recommenders/`, `Evaluation/`, `Utils/`) is **included** in this repository directly from the official QuantumCLEF starter kit.

For QPU execution, [D-Wave Leap](https://cloud.dwavesys.com/leap/) cloud access is required (QPU runs are available only from the official QuantumCLEF workspace).

---

## Dataset

The ICM and URM files are all included in `data/`:

| File | Size | Description |
|---|---|---|
| `data/feature_selection_dataset_100_ICM.csv` | ~1.6 MB | 100-feature ICM (**used in the paper**) |
| `data/feature_selection_dataset_400_ICM.csv` | ~1.9 MB | 400-feature ICM |
| `data/feature_selection_dataset_URM_train.csv` | ~33 MB | User Rating Matrix |

---

## Running the Code

### SA pipeline (reproducible locally)

```bash
python feature_selection_qubo.py \
    --urm_path data/feature_selection_dataset_URM_train.csv \
    --icm_path data/feature_selection_dataset_100_ICM.csv \
    --budget 59 \
    --n_trials 1000
```

The script will:
1. Load and split the data (80/20 outer, 80/20 inner)
2. Compute Random Forest feature importances (R_i)
3. Compute pairwise cosine similarity (S_ij)
4. Run Optuna to tune α, β, λ
5. Report the best feature subset and its NDCG@10

### QPU pipeline

QPU execution is only available within the official QuantumCLEF D-Wave workspace. The notebook `notebooks/qCLEF_submission.ipynb` contains the full submission code as run in that environment.

---

## Citation

If you use this code or paper, please cite:

```bibtex
@inproceedings{skywalkers2026qubo,
  author    = {Bhavishya Prajapat and Alapan Kuila},
  title     = {Balancing Relevance and Redundancy: An Optuna-Tuned {QUBO} Formulation
               for Feature Selection in Recommender Systems},
  booktitle = {Working Notes of {CLEF} 2026},
  series    = {{CEUR} Workshop Proceedings},
  year      = {2026},
  publisher = {CEUR-WS.org}
}
```

Please also cite the QuantumCLEF overview papers:

```bibtex
@inproceedings{overviewquantumclef2026lncs,
  author    = {Andrea Pasin and Maurizio Ferrari Dacrema and Washington Cunha and
               Marcos Andr{\'{e}} Gon{\c{c}}alves and Paolo Cremonesi and Nicola Ferro},
  title     = {Overview of {QuantumCLEF} 2026},
  booktitle = {Experimental {IR} Meets Multilinguality, Multimodality, and Interaction},
  year      = {2026}
}
```

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
