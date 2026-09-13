# COM 837 — ML for Wireless Communication Systems, Assignment 2

**Taral Sanka (IMT2023588)** · International Institute of Information Technology, Bangalore

- **Question 1** — SVM for LOS/NLOS identification.
- **Question 2** — 16-QAM constellation demodulation and clustering with K-means.

---

## Directory structure

```
MLWC_Assignment2/
├── README.md
├── requirements.txt
├── run_all.sh
├── .gitignore
│
├── data/
│   └── .gitkeep
│
├── Q1_LOS_NLOS_SVM/
│   ├── generate_dataset.py
│   ├── question1_svm_los_nlos.py
│   ├── figures/
│   │   ├── partb_accuracy_vs_snr.png
│   │   └── partc_train25_vs_matched.png
│   └── results/
│       ├── partb_accuracy_table.csv
│       ├── partc_accuracy_table.csv
│       └── appendix_kfactor_definitions.csv
│
├── Q2_QAM_KMeans/
│   ├── generate_qam_dataset.py
│   ├── question2_kmeans_qam.py
│   ├── figures/
│   │   ├── partb_elbow_silhouette.png
│   │   ├── partb_clusters_k16_25db.png
│   │   └── partc_adaptive_vs_fixed.png
│   └── results/
│       ├── partb_k_sweep_metrics.csv
│       ├── partb_purity_by_featureset.csv
│       ├── partb_purity_by_featureset_all_snr.csv
│       ├── partc_purity_table.csv
│       ├── partc_centroid_geometry.csv
│       └── appendix_phase_definitions.csv
│
└── report/
    ├── report.tex
    ├── IMT2023588_report.pdf
    ├── figures/
```

### What each part holds

**`data/`** — the generated datasets. These are git-ignored because they are large and fully
reproducible from the fixed random seeds; `run_all.sh` recreates them. `.gitkeep` keeps the
empty directory in the tree.

**`Q1_LOS_NLOS_SVM/`** — everything for Question 1. `generate_dataset.py` is the file provided
with the assignment, committed unmodified. `question1_svm_los_nlos.py` covers Parts (a), (b)
and (c) in one script, plus an appendix diagnostic on the Rician K-factor definition.

**`Q2_QAM_KMeans/`** — everything for Question 2. `generate_qam_dataset.py` synthesises the
16-QAM dataset over AWGN (seed 67); `question2_kmeans_qam.py` covers Parts (a), (b) and (c),
plus an appendix diagnostic on the instantaneous-phase definition.

**`figures/` and `results/`** — written by the analysis scripts. `figures/` holds the PNGs that
appear in the report; `results/` holds the numeric tables behind them, one CSV per table.

**`report/`** — `IMT2023588_report.pdf` is the single PDF submitted to the course portal, containing all
figures and written observations numbered by question. `report.tex` is its source and
`report/figures/` the copies it references. The two `OBSERVATIONS_*.md` files are the longer
working notes the report was condensed from.

**Root files** — `run_all.sh` regenerates every dataset, figure and result table from scratch;
`requirements.txt` lists the four Python packages needed; `.gitignore` excludes the generated
CSVs and LaTeX build artefacts.

---

## Reproducing

```bash
pip install -r requirements.txt
bash run_all.sh                      # datasets, figures, result tables
cd report && pdflatex report.tex && pdflatex report.tex
```

Every script resolves its paths from `__file__`, so it runs from any working directory. All
seeds are fixed (dataset seed 67, `random_state = 42` for splits and K-means), so repeated runs
give identical numbers.
