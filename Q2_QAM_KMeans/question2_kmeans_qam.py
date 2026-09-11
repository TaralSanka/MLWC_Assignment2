"""
=============================================================================
COM 837 - ML for Wireless Communication Systems
Assignment 2, Question 2 : Constellation Demodulation & Clustering (K-means)
=============================================================================

Reads 'qam16_dataset.csv' (written by generate_qam_dataset.py) and runs:

Part (a) : Extracts the two polar features - instantaneous amplitude r and
           instantaneous phase theta - and appends them to the dataframe.

Part (b) : At SNR = 25 dB, sweeps K = 2..20 in Cartesian space and plots
           inertia and silhouette coefficient; fits K = 16 and plots the
           clustered scatter with learned centroids; compares cluster purity
           across three feature sets.

Part (c) : Compares two demodulation strategies at K = 16 across all SNRs -
           an adaptive K-means refitted at every SNR, and a fixed template
           whose centroids are learned once at 25 dB.

Outputs
-------
    ../data/qam16_dataset_with_features.csv     : dataset + r and theta columns
    figures/partb_elbow_silhouette.png          : inertia and silhouette vs K
    figures/partb_clusters_k16_25db.png         : clustered scatter + centroids
    figures/partc_adaptive_vs_fixed.png         : purity vs SNR, both strategies
    results/partb_k_sweep_metrics.csv           : inertia/silhouette per K
    results/partb_purity_by_featureset*.csv     : purity of the three feature sets
    results/partc_purity_table.csv              : Part (c) numbers for the report
    results/partc_centroid_geometry.csv         : centroid-drift diagnostic
    results/appendix_phase_definitions.csv      : arctan vs arctan2 diagnostic
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless backend - write PNGs only
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Repository paths - every input and output is resolved relative to this file,
# so the script runs correctly from any working directory.
# -----------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR.parent / "data"      # shared, git-ignored generated CSVs
FIGURE_DIR  = BASE_DIR / "figures"          # committed PNG figures
RESULTS_DIR = BASE_DIR / "results"          # committed numeric result tables

for _directory in (DATA_DIR, FIGURE_DIR, RESULTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Global experiment settings
# -----------------------------------------------------------------------------
DATASET_FILE        = DATA_DIR / "qam16_dataset.csv"
NUM_CONSTELLATION_PTS = 16          # true number of 16-QAM symbols
K_SWEEP_RANGE       = range(2, 21)  # K = 2 .. 20 inclusive
KMEANS_INIT         = "k-means++"   # initialisation required by the assignment
KMEANS_N_INIT       = 10            # restarts, required by the assignment
KMEANS_RANDOM_STATE = 42            # fixed so results are reproducible
REFERENCE_SNR_DB    = 25            # SNR used for Part (b) and for the template


def make_kmeans(n_clusters):
    """Factory returning a K-means object with the required settings."""
    return KMeans(
        n_clusters=n_clusters,
        init=KMEANS_INIT,
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
    )


# =============================================================================
# PART (a) : POLAR FEATURE EXTRACTION
# =============================================================================

def load_dataset(dataset_file=DATASET_FILE):
    """Read the CSV written by generate_qam_dataset.py."""
    dataframe = pd.read_csv(dataset_file)
    print(f"Loaded {dataset_file} : {len(dataframe)} rows, "
          f"{dataframe['snr_db'].nunique()} SNR values")
    return dataframe


def add_polar_features(dataframe):
    """
    Part (a): append the instantaneous amplitude and phase of every received
    sample.

        r     = sqrt(rx_I^2 + rx_Q^2)          Eq. (1)
        theta = angle of (rx_I + j*rx_Q)       Eq. (2), in radians

    Note on the phase: Eq. (2) is written as arctan(rx_Q / rx_I), but the
    two-argument form np.arctan2(rx_Q, rx_I) is used here. Plain arctan
    returns values in (-pi/2, pi/2) only, so it maps a point and its negation
    to the *same* phase - the symbol at (+3,+3) and the symbol at (-3,-3)
    would become indistinguishable, collapsing the constellation onto half
    the plane and destroying the clustering before it starts. arctan2 keeps
    all four quadrants distinct and is the standard definition of
    instantaneous phase. The consequence of using plain arctan is quantified
    in the appendix diagnostic at the bottom of this file.
    """
    dataframe["r"]     = np.hypot(dataframe["rx_I"], dataframe["rx_Q"])
    dataframe["theta"] = np.arctan2(dataframe["rx_Q"], dataframe["rx_I"])

    print("\nPart (a): added polar feature columns -> ['r', 'theta']")
    print(f"  r     range: [{dataframe['r'].min():.3f}, {dataframe['r'].max():.3f}]")
    print(f"  theta range: [{dataframe['theta'].min():.3f}, "
          f"{dataframe['theta'].max():.3f}] rad")

    return dataframe


# =============================================================================
# CLUSTER PURITY
# =============================================================================

def cluster_purity(cluster_labels, true_labels):
    """
    Cluster purity: assign each cluster to the ground-truth symbol that is
    most frequent inside it, then return the overall fraction of samples that
    end up correctly assigned.

        purity = (1/N) * sum_over_clusters( max_over_symbols count(k, symbol) )

    A contingency table (clusters x symbols) is built with pandas.crosstab,
    the row-wise maximum gives each cluster's majority vote, and the sum of
    those maxima over N is the purity. Note that two clusters are allowed to
    claim the same symbol - that is what makes purity penalise the merged
    clusters seen at low SNR.
    """
    contingency_table = pd.crosstab(cluster_labels, true_labels)
    majority_counts   = contingency_table.max(axis=1).sum()

    return majority_counts / len(true_labels)


# =============================================================================
# PART (b) : MODEL SELECTION, CLUSTER VISUALISATION, FEATURE-SET COMPARISON
# =============================================================================

def run_k_sweep(cartesian_features):
    """
    Fit K-means for K = 2..20 on the Cartesian data and record the two model
    selection metrics.

    Inertia (within-cluster sum of squares) falls monotonically with K by
    construction, so it is read for its *elbow*, not its minimum. The
    silhouette coefficient balances within-cluster tightness against
    between-cluster separation and does have a meaningful maximum.

    Returns a dataframe indexed by K.
    """
    sweep_records = []

    for n_clusters in K_SWEEP_RANGE:
        kmeans_model   = make_kmeans(n_clusters)
        cluster_labels = kmeans_model.fit_predict(cartesian_features)

        sweep_records.append({
            "K":          n_clusters,
            "inertia":    kmeans_model.inertia_,
            "silhouette": silhouette_score(cartesian_features, cluster_labels),
        })

    sweep_results = pd.DataFrame(sweep_records).set_index("K")

    best_silhouette_k = sweep_results["silhouette"].idxmax()
    print(f"\nPart (b1): K sweep complete. "
          f"Silhouette peaks at K = {best_silhouette_k} "
          f"({sweep_results['silhouette'].max():.4f})")

    return sweep_results


def plot_k_sweep(sweep_results, output_file=None):
    """Side-by-side inertia and silhouette curves, with K = 16 marked."""
    output_file = output_file or FIGURE_DIR / "partb_elbow_silhouette.png"

    figure, (inertia_axis, silhouette_axis) = plt.subplots(1, 2, figsize=(13, 5))

    inertia_axis.plot(sweep_results.index, sweep_results["inertia"],
                      marker="o", color="tab:blue")
    inertia_axis.axvline(NUM_CONSTELLATION_PTS, color="tab:red",
                         linestyle="--", alpha=0.7, label="K = 16 (true symbol count)")
    inertia_axis.set_xlabel("Number of clusters K")
    inertia_axis.set_ylabel("Within-cluster sum of squares (inertia)")
    inertia_axis.set_title("Elbow curve")
    inertia_axis.set_xticks(list(K_SWEEP_RANGE)[::2])
    inertia_axis.grid(True, alpha=0.3)
    inertia_axis.legend()

    silhouette_axis.plot(sweep_results.index, sweep_results["silhouette"],
                         marker="s", color="tab:green")
    silhouette_axis.axvline(NUM_CONSTELLATION_PTS, color="tab:red",
                            linestyle="--", alpha=0.7, label="K = 16 (true symbol count)")
    silhouette_axis.set_xlabel("Number of clusters K")
    silhouette_axis.set_ylabel("Silhouette coefficient")
    silhouette_axis.set_title("Silhouette curve")
    silhouette_axis.set_xticks(list(K_SWEEP_RANGE)[::2])
    silhouette_axis.grid(True, alpha=0.3)
    silhouette_axis.legend()

    figure.suptitle(f"Part (b): K-means model selection on Cartesian features, "
                    f"SNR = {REFERENCE_SNR_DB} dB")
    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    print(f"Saved figure: {output_file}")


def plot_clusters_k16(cartesian_features, reference_subset,
                      output_file=None):
    """
    Fit K = 16 on the Cartesian data at the reference SNR and scatter-plot the
    received samples coloured by assigned cluster, with the learned centroids
    overlaid and the true transmitted constellation shown for comparison.

    Returns the fitted model so the caller can reuse it.
    """
    output_file = output_file or FIGURE_DIR / "partb_clusters_k16_25db.png"

    kmeans_model   = make_kmeans(NUM_CONSTELLATION_PTS)
    cluster_labels = kmeans_model.fit_predict(cartesian_features)
    centroids      = kmeans_model.cluster_centers_

    figure, axis = plt.subplots(figsize=(8, 8))

    axis.scatter(cartesian_features[:, 0], cartesian_features[:, 1],
                 c=cluster_labels, cmap="tab20", s=8, alpha=0.6,
                 label="Received samples (coloured by cluster)")

    # learned centroids
    axis.scatter(centroids[:, 0], centroids[:, 1],
                 marker="X", s=220, c="black", edgecolors="white", linewidths=1.5,
                 zorder=3, label="Learned K-means centroids")

    # true transmitted constellation, for visual reference
    true_points = reference_subset.drop_duplicates("symbol_id")
    axis.scatter(true_points["tx_I"], true_points["tx_Q"],
                 marker="+", s=160, c="red", linewidths=2,
                 zorder=4, label="True constellation points")

    axis.set_xlabel("In-phase component  $Rx_I$")
    axis.set_ylabel("Quadrature component  $Rx_Q$")
    axis.set_title(f"Part (b): K-means with K = 16 on Cartesian features, "
                   f"SNR = {REFERENCE_SNR_DB} dB")
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    print(f"Saved figure: {output_file}")

    purity = cluster_purity(cluster_labels, reference_subset["symbol_id"].to_numpy())
    print(f"Part (b3): purity of the K = 16 Cartesian fit at "
          f"{REFERENCE_SNR_DB} dB = {purity:.4f}")

    return kmeans_model


def compare_feature_sets(reference_subset):
    """
    Part (b4): compute cluster purity at K = 16 for the three feature sets.

        Set 1 : (rx_I, rx_Q)              - raw Cartesian, unscaled
        Set 2 : (r, theta)                - raw polar, unscaled
        Set 3 : (rx_I, rx_Q, r, theta)    - all four, StandardScaler applied

    Only Set 3 is scaled, as specified in the assignment. Sets 1 and 2 are
    left in their native units on purpose: Set 2's distortion comes precisely
    from mixing an amplitude in volts with a phase in radians on one Euclidean
    axis, and scaling it would hide the effect the question is asking about.
    """
    true_symbols = reference_subset["symbol_id"].to_numpy()

    feature_set_definitions = {
        "Set 1: (rx_I, rx_Q)":            (["rx_I", "rx_Q"],            False),
        "Set 2: (r, theta)":              (["r", "theta"],              False),
        "Set 3: (rx_I, rx_Q, r, theta)":  (["rx_I", "rx_Q", "r", "theta"], True),
    }

    purity_records = {}
    for set_name, (column_names, needs_scaling) in feature_set_definitions.items():
        feature_matrix = reference_subset[column_names].to_numpy()

        if needs_scaling:
            feature_matrix = StandardScaler().fit_transform(feature_matrix)

        cluster_labels = make_kmeans(NUM_CONSTELLATION_PTS).fit_predict(feature_matrix)
        purity_records[set_name] = cluster_purity(cluster_labels, true_symbols)

    results = pd.Series(purity_records, name="cluster_purity").to_frame()

    print(f"\nPart (b4): cluster purity at K = 16, SNR = {REFERENCE_SNR_DB} dB")
    print(results.to_string(float_format=lambda value: f"{value:8.4f}"))
    print(f"  Highest purity: {results['cluster_purity'].idxmax()}")

    return results


def compare_feature_sets_across_snr(dataframe):
    """
    Supplementary to Part (b4).

    At SNR = 25 dB all three feature sets reach purity 1.0000, so the required
    comparison is saturated and cannot rank them. Repeating it at every SNR
    breaks the tie: the differences between the representations only appear
    once the clusters are close enough together for the distance metric to
    matter.
    """
    feature_set_definitions = {
        "Set 1: (rx_I, rx_Q)":            (["rx_I", "rx_Q"],               False),
        "Set 2: (r, theta)":              (["r", "theta"],                 False),
        "Set 3: (rx_I, rx_Q, r, theta)":  (["rx_I", "rx_Q", "r", "theta"], True),
    }

    purity_records = {}
    for snr_db in sorted(dataframe["snr_db"].unique()):
        snr_subset   = dataframe[dataframe["snr_db"] == snr_db]
        true_symbols = snr_subset["symbol_id"].to_numpy()

        row = {}
        for set_name, (column_names, needs_scaling) in feature_set_definitions.items():
            feature_matrix = snr_subset[column_names].to_numpy()
            if needs_scaling:
                feature_matrix = StandardScaler().fit_transform(feature_matrix)

            cluster_labels = make_kmeans(NUM_CONSTELLATION_PTS).fit_predict(feature_matrix)
            row[set_name]  = cluster_purity(cluster_labels, true_symbols)

        purity_records[snr_db] = row

    results = pd.DataFrame(purity_records).T
    results.index.name = "snr_db"

    print("\nPart (b4, supplementary): cluster purity of the three feature sets vs SNR")
    print(results.to_string(float_format=lambda value: f"{value:8.4f}"))

    return results


def analyse_centroid_geometry(dataframe, template_model):
    """
    Supplementary to Part (c): quantifies centroid merging.

    For each SNR, the adaptive centroids are compared against the true
    (noiseless) constellation grid, and against each other:

      - rms_centroid_error : RMS distance from each adaptive centroid to its
        nearest true constellation point. Large values mean the fitted
        centroids have drifted off the grid.
      - min_centroid_gap : smallest distance between any two adaptive
        centroids. Small values mean two centroids have collapsed onto the
        same blob - i.e. merging - while some other true symbol is left
        without a centroid at all.

    The fixed template's values are reported once for comparison, since by
    construction they do not change with SNR.
    """
    true_constellation = (dataframe.drop_duplicates("symbol_id")[["tx_I", "tx_Q"]]
                          .to_numpy())

    def describe(centroids):
        distances_to_grid = np.linalg.norm(
            centroids[:, None, :] - true_constellation[None, :, :], axis=2
        )
        rms_error = np.sqrt((distances_to_grid.min(axis=1) ** 2).mean())

        pairwise = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        return rms_error, pairwise.min()

    geometry_records = {}
    for snr_db in sorted(dataframe["snr_db"].unique()):
        snr_subset         = dataframe[dataframe["snr_db"] == snr_db]
        cartesian_features = snr_subset[["rx_I", "rx_Q"]].to_numpy()

        adaptive_centroids = make_kmeans(NUM_CONSTELLATION_PTS)\
            .fit(cartesian_features).cluster_centers_

        rms_error, min_gap = describe(adaptive_centroids)
        geometry_records[snr_db] = {
            "adaptive rms error to true grid": rms_error,
            "adaptive min centroid gap":       min_gap,
        }

    template_rms, template_gap = describe(template_model.cluster_centers_)

    results = pd.DataFrame(geometry_records).T
    results.index.name = "snr_db"
    results["template rms error to true grid"] = template_rms
    results["template min centroid gap"]       = template_gap

    print("\nPart (c, supplementary): adaptive centroid geometry vs the true grid")
    print(results.to_string(float_format=lambda value: f"{value:8.4f}"))
    print(f"  (true grid minimum symbol spacing = "
          f"{2 / np.sqrt(10):.4f})")

    return results


# =============================================================================
# PART (c) : ADAPTIVE K-MEANS VS FIXED 25 dB TEMPLATE
# =============================================================================

def run_part_c(dataframe, template_model):
    """
    Compare the two demodulation strategies at K = 16 across all SNRs.

    Strategy 1 (Adaptive)      : refit K-means on each SNR's own Cartesian
                                 data and score its purity there.
    Strategy 2 (Fixed template): reuse the centroids learned once at
                                 REFERENCE_SNR_DB and only *predict* at every
                                 SNR - no refitting, so the geometry of the
                                 known 16-QAM grid is preserved.
    """
    purity_records = {}

    for snr_db in sorted(dataframe["snr_db"].unique()):
        snr_subset         = dataframe[dataframe["snr_db"] == snr_db]
        cartesian_features = snr_subset[["rx_I", "rx_Q"]].to_numpy()
        true_symbols       = snr_subset["symbol_id"].to_numpy()

        adaptive_labels = make_kmeans(NUM_CONSTELLATION_PTS).fit_predict(cartesian_features)
        fixed_labels    = template_model.predict(cartesian_features)

        purity_records[snr_db] = {
            "Adaptive K-Means": cluster_purity(adaptive_labels, true_symbols),
            "Fixed Template":   cluster_purity(fixed_labels,    true_symbols),
        }

    results = pd.DataFrame(purity_records).T
    results.index.name = "snr_db"
    results["Difference (fixed - adaptive)"] = (
        results["Fixed Template"] - results["Adaptive K-Means"]
    )

    print("\nPart (c): cluster purity vs SNR")
    print(results.to_string(float_format=lambda value: f"{value:8.4f}"))

    return results


def plot_part_c(results, output_file=None):
    """Two-curve figure comparing the adaptive and fixed-template strategies."""
    output_file = output_file or FIGURE_DIR / "partc_adaptive_vs_fixed.png"

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.plot(results.index, results["Adaptive K-Means"],
              marker="o", linewidth=2.4, color="tab:blue",
              label="Adaptive K-Means (refitted per SNR)")

    axis.plot(results.index, results["Fixed Template"],
              marker="s", linewidth=2.4, color="tab:red", linestyle="--",
              label=f"Fixed Template (centroids from {REFERENCE_SNR_DB} dB)")

    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Cluster purity")
    axis.set_title("Part (c): 16-QAM demodulation by clustering\n"
                   "adaptive K-means vs fixed 25 dB centroid template (K = 16)")
    axis.set_xticks(results.index)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right", fontsize=10)

    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    print(f"Saved figure: {output_file}")


# =============================================================================
# APPENDIX : DIAGNOSTIC FOR THE PART (a) PHASE DEFINITION
# =============================================================================

def run_phase_definition_check(reference_subset):
    """
    Quantifies why arctan2 is used for the instantaneous phase instead of the
    literal arctan(rx_Q / rx_I).

    Plain arctan folds quadrant 3 onto quadrant 1 and quadrant 4 onto
    quadrant 2, so half of the 16 symbols become aliases of the other half in
    the (r, theta) plane. Purity on Feature Set 2 is recomputed with both
    phase definitions to show the size of the effect.
    """
    true_symbols = reference_subset["symbol_id"].to_numpy()

    two_quadrant_phase = np.arctan(reference_subset["rx_Q"] / reference_subset["rx_I"])

    phase_variants = {
        "arctan2(Q, I)  (used)":     reference_subset["theta"].to_numpy(),
        "arctan(Q / I)  (literal)":  two_quadrant_phase.to_numpy(),
    }

    purity_records = {}
    for variant_name, phase_values in phase_variants.items():
        polar_features = np.column_stack([reference_subset["r"].to_numpy(), phase_values])
        cluster_labels = make_kmeans(NUM_CONSTELLATION_PTS).fit_predict(polar_features)
        purity_records[variant_name] = cluster_purity(cluster_labels, true_symbols)

    results = pd.Series(purity_records, name="cluster_purity").to_frame()

    print("\nAppendix: Feature Set 2 purity under the two phase definitions")
    print(results.to_string(float_format=lambda value: f"{value:8.4f}"))

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ---- Part (a) -----------------------------------------------------------
    dataframe = add_polar_features(load_dataset())
    dataframe.to_csv(DATA_DIR / "qam16_dataset_with_features.csv", index=False)
    print("Saved dataset with features: qam16_dataset_with_features.csv")

    # The reference subset used throughout Part (b)
    reference_subset   = dataframe[dataframe["snr_db"] == REFERENCE_SNR_DB].reset_index(drop=True)
    cartesian_features = reference_subset[["rx_I", "rx_Q"]].to_numpy()

    # ---- Part (b) -----------------------------------------------------------
    sweep_results = run_k_sweep(cartesian_features)
    sweep_results.to_csv(RESULTS_DIR / "partb_k_sweep_metrics.csv")
    plot_k_sweep(sweep_results)

    template_model = plot_clusters_k16(cartesian_features, reference_subset)

    feature_set_results = compare_feature_sets(reference_subset)
    feature_set_results.to_csv(RESULTS_DIR / "partb_purity_by_featureset.csv")

    feature_set_vs_snr = compare_feature_sets_across_snr(dataframe)
    feature_set_vs_snr.to_csv(RESULTS_DIR / "partb_purity_by_featureset_all_snr.csv")

    # ---- Part (c) -----------------------------------------------------------
    part_c_results = run_part_c(dataframe, template_model)
    part_c_results.to_csv(RESULTS_DIR / "partc_purity_table.csv")
    plot_part_c(part_c_results)

    centroid_geometry = analyse_centroid_geometry(dataframe, template_model)
    centroid_geometry.to_csv(RESULTS_DIR / "partc_centroid_geometry.csv")

    # ---- Appendix -----------------------------------------------------------
    phase_check = run_phase_definition_check(reference_subset)
    phase_check.to_csv(RESULTS_DIR / "appendix_phase_definitions.csv")


if __name__ == "__main__":
    main()
