# Dataset

This directory contains the item feature matrices for QuantumCLEF Task 1B.

## Included Files

| File | Size | Description |
|---|---|---|
| `feature_selection_dataset_100_ICM.csv` | ~1.6 MB | 100-feature Item Content Matrix (used in the paper) |
| `feature_selection_dataset_400_ICM.csv` | ~1.9 MB | 400-feature Item Content Matrix (alternative, not used) |

## Missing: URM (User Rating Matrix)

The file `feature_selection_dataset_URM_train.csv` (~35 MB) is **not included** in this repository due to its size.

**Download it from the official QuantumCLEF repository:**
> 📌 Link pending — the organizers will release this at the official qCLEF GitHub repo.

Once downloaded, place it here:
```
data/feature_selection_dataset_URM_train.csv
```

## Dataset Statistics

| Stat | Value |
|---|---|
| Users | 20,428 |
| Items | 14,607 |
| Interactions (URM nnz) | 3,209,730 |
| URM density | ~0.54% |
| ICM features (100_ICM) | 100 |
| ICM non-zeros | 106,979 |
| ICM density | ~0.07% |
