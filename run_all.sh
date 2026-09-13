#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Regenerate every dataset, figure and result table in this repository.
# Safe to re-run: all random seeds are fixed, so output is deterministic.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
mkdir -p data

echo "=== Question 1: generating LOS/NLOS dataset ==================="
# generate_dataset.py is used exactly as provided and writes to its working
# directory, so it is invoked from inside data/ rather than being modified.
( cd data && python "$REPO_ROOT/Q1_LOS_NLOS_SVM/generate_dataset.py" )

echo "=== Question 1: SVM feature analysis =========================="
python Q1_LOS_NLOS_SVM/question1_svm_los_nlos.py

echo "=== Question 2: generating 16-QAM dataset ====================="
python Q2_QAM_KMeans/generate_qam_dataset.py

echo "=== Question 2: K-means clustering analysis ==================="
python Q2_QAM_KMeans/question2_kmeans_qam.py

echo
echo "Done. Figures in Q*/figures/, numeric tables in Q*/results/."
