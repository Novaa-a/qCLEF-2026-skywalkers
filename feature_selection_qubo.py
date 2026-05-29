"""
QUBO-based Feature Selection for Recommender Systems
=====================================================
QuantumCLEF 2026 Task 1B — Team Skywalkers

This script implements the core Optuna-tuned QUBO feature selection pipeline
using Simulated Annealing (SA). QPU execution requires the D-Wave Leap cloud
platform and the official QuantumCLEF workspace.

Usage:
    python feature_selection_qubo.py \\
        --urm_path data/feature_selection_dataset_URM_train.csv \\
        --icm_path data/feature_selection_dataset_100_ICM.csv \\
        --budget 59 \\
        --n_trials 1000

Requirements:
    - The qCLEF framework (ItemKNNCBFRecommender, EvaluatorHoldout) must be
      on the Python path. This framework will be released by the QuantumCLEF
      organizers. See README.md for the link.
    - pip install -r requirements.txt
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Add the bundled qCLEF framework (Recommenders/, Evaluation/, Utils/) to the path.
# These are included in this repository from the official QuantumCLEF starter kit.
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Data loading and matrix construction
# ---------------------------------------------------------------------------

def load_data(urm_path: str, icm_path: str):
    """Load URM and ICM from CSV and return sparse matrices."""
    urm_df = pd.read_csv(urm_path)
    icm_df = pd.read_csv(icm_path)

    n_users = int(urm_df["UserID"].max()) + 1
    n_items = int(max(urm_df["ItemID"].max(), icm_df["ItemID"].max())) + 1
    n_features = int(icm_df["FeatureID"].max()) + 1

    urm_sparse = sps.csr_matrix(
        (np.ones(len(urm_df), dtype=np.float32),
         (urm_df["UserID"].values, urm_df["ItemID"].values)),
        shape=(n_users, n_items),
    )
    urm_sparse.sum_duplicates()
    urm_sparse.data[:] = 1.0

    icm_sparse = sps.csr_matrix(
        (icm_df["Value"].astype(np.float32),
         (icm_df["ItemID"].values, icm_df["FeatureID"].values)),
        shape=(n_items, n_features),
    )
    icm_sparse.sum_duplicates()

    print(f"URM: {urm_sparse.shape}, nnz={urm_sparse.nnz}")
    print(f"ICM: {icm_sparse.shape}, nnz={icm_sparse.nnz}")
    print(f"Users={n_users}, Items={n_items}, Features={n_features}")
    return urm_sparse, icm_sparse, urm_df


def build_urm(df: pd.DataFrame, shape: tuple) -> sps.csr_matrix:
    """Construct a binary URM from a DataFrame of (UserID, ItemID) rows."""
    urm = sps.csr_matrix(
        (np.ones(len(df), dtype=np.float32),
         (df["UserID"].values, df["ItemID"].values)),
        shape=shape,
    )
    urm.sum_duplicates()
    urm.data[:] = 1.0
    return urm


def split_data(urm_df: pd.DataFrame, urm_shape: tuple):
    """
    Two-tier split to prevent leakage:
      - Outer 80/20: search_df / final_df  (final holdout never used for tuning)
      - Inner 80/20 of search: train_opt / val_opt  (guides Optuna)
    """
    search_df, final_df = train_test_split(
        urm_df, test_size=0.20, random_state=42, shuffle=True
    )
    train_opt_df, val_opt_df = train_test_split(
        search_df, test_size=0.20, random_state=1042, shuffle=True
    )
    return (
        build_urm(train_opt_df, urm_shape),
        build_urm(val_opt_df, urm_shape),
        build_urm(search_df, urm_shape),
        build_urm(final_df, urm_shape),
    )


# ---------------------------------------------------------------------------
# QUBO coefficient computation
# ---------------------------------------------------------------------------

def compute_relevance(urm_train: sps.csr_matrix, icm: sps.csr_matrix) -> np.ndarray:
    """
    Compute per-feature relevance scores R_i using a Random Forest.

    The RF is trained to predict user-item interactions from item features.
    Gini importance across all trees gives the linear QUBO coefficients.

    Returns
    -------
    R : np.ndarray, shape (n_features,)
        Normalized feature importance scores in [0, 1].
    """
    # Build a feature matrix for items that have at least one interaction
    interacted_items = np.unique(urm_train.nonzero()[1])
    X = icm[interacted_items].toarray()

    # Binary label: 1 if item appears more than median times, else 0
    interaction_counts = np.asarray(urm_train[:, interacted_items].sum(axis=0)).ravel()
    y = (interaction_counts > np.median(interaction_counts)).astype(int)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    R = rf.feature_importances_
    # Normalize to [0, 1]
    if R.max() > 0:
        R = R / R.max()
    return R


def compute_redundancy(icm: sps.csr_matrix) -> np.ndarray:
    """
    Compute pairwise feature redundancy S_ij via cosine similarity on ICM columns.

    Returns
    -------
    S : np.ndarray, shape (n_features, n_features)
        Symmetric matrix where S[i, j] in [0, 1] measures overlap between features.
    """
    # Transpose so rows = features, then compute pairwise cosine similarity
    feature_matrix = icm.T.toarray()  # shape: (n_features, n_items)
    S = cosine_similarity(feature_matrix)
    np.fill_diagonal(S, 0.0)  # self-similarity is irrelevant
    return S


# ---------------------------------------------------------------------------
# QUBO matrix construction
# ---------------------------------------------------------------------------

def build_qubo_matrix(
    R: np.ndarray,
    S: np.ndarray,
    k: int,
    alpha: float,
    beta: float,
    lam: float,
) -> np.ndarray:
    """
    Construct the QUBO matrix Q for the feature selection objective:

        E(x) = -alpha * sum_i R_i * x_i
               + beta  * sum_{i<j} S_ij * x_i * x_j
               + lam   * (sum_i x_i - k)^2

    The annealer seeks x* = argmin x^T Q x.

    Parameters
    ----------
    R     : relevance vector, shape (N,)
    S     : redundancy matrix, shape (N, N)
    k     : target feature budget
    alpha : relevance weight (>= 0)
    beta  : redundancy weight (>= 0)
    lam   : budget constraint weight (>= 0)

    Returns
    -------
    Q : np.ndarray, shape (N, N)  — upper-triangular QUBO matrix
    """
    N = len(R)
    Q = np.zeros((N, N), dtype=np.float64)

    # Diagonal: relevance reward + budget constraint linear term
    for i in range(N):
        Q[i, i] = -alpha * R[i] + lam * (1 - 2 * k)

    # Off-diagonal: redundancy penalty + budget constraint quadratic term
    for i in range(N):
        for j in range(i + 1, N):
            Q[i, j] = beta * S[i, j] + 2 * lam

    return Q


# ---------------------------------------------------------------------------
# Simulated Annealing solver (via dwave-neal)
# ---------------------------------------------------------------------------

def solve_qubo_sa(Q: np.ndarray, num_reads: int = 250) -> np.ndarray:
    """
    Solve the QUBO using digital Simulated Annealing via dwave-neal.

    Parameters
    ----------
    Q         : QUBO matrix, shape (N, N)
    num_reads : number of independent annealing runs

    Returns
    -------
    x : np.ndarray of shape (N,) with values in {0, 1} — the best valid solution
    """
    try:
        import neal
    except ImportError:
        raise ImportError(
            "dwave-neal is required for SA execution.\n"
            "Install it with: pip install dwave-neal"
        )

    N = Q.shape[0]
    qubo_dict = {(i, j): Q[i, j] for i in range(N) for j in range(i, N) if Q[i, j] != 0}

    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo_dict, num_reads=num_reads)

    # Return the lowest-energy sample
    best_sample = response.first.sample
    return np.array([best_sample[i] for i in range(N)], dtype=int)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(R, S, k, icm_sparse, urm_train, evaluator):
    """
    Returns an Optuna objective function that:
      1. Receives alpha, beta, lambda from the TPE sampler
      2. Builds the QUBO matrix and solves it with SA
      3. Evaluates the resulting feature subset with Item-KNN
      4. Returns NDCG@10 as the maximization target
    """
    try:
        from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    except ImportError:
        raise ImportError(
            "ItemKNNCBFRecommender is not available. "
            "This class is part of the official qCLEF framework, "
            "which will be released by the QuantumCLEF organizers. "
            "See README.md for the link."
        )

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.1, 5.0)
        beta  = trial.suggest_float("beta",  0.0, 2.0)
        lam   = trial.suggest_float("lam",   0.1, 10.0)

        Q = build_qubo_matrix(R, S, k, alpha, beta, lam)
        x = solve_qubo_sa(Q, num_reads=250)

        n_selected = int(x.sum())
        if n_selected != k:
            # Penalize solutions that don't meet the exact budget
            return 0.0

        selected_features = np.where(x == 1)[0]
        icm_reduced = icm_sparse[:, selected_features]

        rec = ItemKNNCBFRecommender(urm_train, icm_reduced)
        rec.fit(topK=100, shrink=5, similarity="cosine", normalize=True, feature_weighting="none")

        result_df, _ = evaluator.evaluateRecommender(rec)
        ndcg = float(result_df.loc[10, "NDCG"])
        return ndcg

    return objective


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(urm_path: str, icm_path: str, budget: int, n_trials: int):
    """End-to-end Optuna-QUBO-SA feature selection pipeline."""

    print("=" * 60)
    print("QUBO Feature Selection — Skywalkers @ QuantumCLEF 2026")
    print("=" * 60)

    # 1. Load data
    urm_sparse, icm_sparse, urm_df = load_data(urm_path, icm_path)

    # 2. Split
    URM_train_opt, URM_val_opt, URM_search, URM_final = split_data(
        urm_df, urm_sparse.shape
    )

    # Try importing the official evaluator
    try:
        from Evaluation.Evaluator import EvaluatorHoldout
        evaluator_opt   = EvaluatorHoldout(URM_val_opt, cutoff_list=[10])
        evaluator_final = EvaluatorHoldout(URM_final,   cutoff_list=[10])
    except ImportError:
        raise ImportError(
            "EvaluatorHoldout is not available. "
            "This class is part of the official qCLEF framework. "
            "See README.md for the link."
        )

    # 3. Compute QUBO coefficients
    print(f"\n[1/3] Computing QUBO coefficients for k={budget}...")
    R = compute_relevance(URM_train_opt, icm_sparse)
    S = compute_redundancy(icm_sparse)
    print(f"  R: min={R.min():.4f}, max={R.max():.4f}, mean={R.mean():.4f}")
    print(f"  S: mean off-diagonal={S[S > 0].mean():.4f}")

    # 4. Optuna search
    print(f"\n[2/3] Running Optuna ({n_trials} trials) ...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    objective = make_objective(R, S, budget, icm_sparse, URM_train_opt, evaluator_opt)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    print(f"\n  Best tune NDCG@10 : {best_trial.value:.6f}")
    print(f"  Best params       : alpha={best_trial.params['alpha']:.4f}, "
          f"beta={best_trial.params['beta']:.4f}, lam={best_trial.params['lam']:.4f}")

    # 5. Evaluate best subset on final holdout
    print(f"\n[3/3] Evaluating best subset on final holdout ...")
    best_alpha = best_trial.params["alpha"]
    best_beta  = best_trial.params["beta"]
    best_lam   = best_trial.params["lam"]

    Q_best = build_qubo_matrix(R, S, budget, best_alpha, best_beta, best_lam)
    x_best = solve_qubo_sa(Q_best, num_reads=1000)

    n_selected = int(x_best.sum())
    selected_features = np.where(x_best == 1)[0]
    print(f"  Selected {n_selected} features (target: {budget})")
    print(f"  Feature indices: {selected_features.tolist()}")

    from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    icm_reduced = icm_sparse[:, selected_features]
    rec_final = ItemKNNCBFRecommender(URM_search, icm_reduced)
    rec_final.fit(topK=100, shrink=5, similarity="cosine", normalize=True, feature_weighting="none")

    result_df, _ = evaluator_final.evaluateRecommender(rec_final)
    ndcg_final = float(result_df.loc[10, "NDCG"])

    print(f"\n  Final holdout NDCG@10: {ndcg_final:.6f}")
    print("=" * 60)
    return selected_features, ndcg_final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna-QUBO feature selection for QuantumCLEF Task 1B"
    )
    parser.add_argument("--urm_path", required=True, help="Path to URM CSV file")
    parser.add_argument("--icm_path", required=True, help="Path to ICM CSV file")
    parser.add_argument("--budget",   type=int, default=59, help="Feature budget k (default: 59)")
    parser.add_argument("--n_trials", type=int, default=1000, help="Optuna trials (default: 1000)")
    args = parser.parse_args()

    run_pipeline(args.urm_path, args.icm_path, args.budget, args.n_trials)
