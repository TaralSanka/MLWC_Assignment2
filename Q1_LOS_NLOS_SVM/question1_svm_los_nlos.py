"""
=============================================================================
COM 837 - ML for Wireless Communication Systems
Assignment 2, Question 1 : SVM for LOS/NLOS Identification
=============================================================================

Reference:
    C. Huang, A. F. Molisch, R. He, R. Wang, P. Tang, B. Ai and Z. Zhong,
    "Machine Learning-Enabled LOS/NLOS Identification for MIMO Systems in
    Dynamic Environments," IEEE Trans. Wireless Commun., vol. 19, no. 6,
    pp. 3643-3657, Jun. 2020.

What this script does
---------------------
Part (a) : Extracts the five scalar channel features listed in the
           assignment from every channel realisation in
           'los_nlos_dataset.csv' (produced by the *unmodified*
           generate_dataset.py) and appends them to the dataframe.

Part (b) : Trains six RBF-kernel SVMs (C = 1) per SNR value - five
           single-feature classifiers and one that uses all five features -
           and plots classification accuracy vs SNR.

Part (c) : Trains the five-feature SVM once on SNR = 25 dB data only and
           evaluates that single fixed model at every SNR, comparing it
           against the matched "train = test SNR" curve from Part (b).

Outputs
-------
    ../data/los_nlos_dataset_with_features.csv  : dataset + the five features
    figures/partb_accuracy_vs_snr.png           : Part (b) figure
    figures/partc_train25_vs_matched.png        : Part (c) figure
    results/partb_accuracy_table.csv            : Part (b) numbers for the report
    results/partc_accuracy_table.csv            : Part (c) numbers for the report
    results/appendix_kfactor_definitions.csv    : K-factor definition diagnostic
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless backend - write PNGs only
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
# Global experiment settings (kept in one place so nothing is hard-coded twice)
# -----------------------------------------------------------------------------
DATASET_FILE      = DATA_DIR / "los_nlos_dataset.csv"   # from generate_dataset.py
NUM_TAPS          = 6                        # L : taps per channel realisation
TEST_FRACTION     = 0.20                     # 80/20 train-test split
SPLIT_RANDOM_SEED = 42                       # fixed so splits are reproducible
SVM_C             = 1.0                      # regularisation constant, C = 1
SVM_KERNEL        = "rbf"                    # radial basis function kernel
FIXED_TRAIN_SNR   = 25                       # SNR used for the Part (c) model

# The five features extracted in Part (a), in the order used everywhere below.
FEATURE_NAMES = [
    "kurtosis",          # peakedness of the tap-amplitude distribution
    "skewness",          # asymmetry of the tap-amplitude distribution
    "rising_time",       # delay gap between first tap and strongest tap
    "rms_delay_spread",  # second central moment of the power delay profile
    "rician_k_factor",   # dominant-tap power relative to amplitude variance
]

# Human-readable labels used in the plot legends.
FEATURE_LABELS = {
    "kurtosis":         "SVM-1: Kurtosis",
    "skewness":         "SVM-2: Skewness",
    "rising_time":      "SVM-3: Rising time",
    "rms_delay_spread": "SVM-4: RMS delay spread",
    "rician_k_factor":  "SVM-5: Rician K-factor",
}


# =============================================================================
# PART (a) : FEATURE EXTRACTION
# =============================================================================

def load_raw_dataset(dataset_file=DATASET_FILE):
    """Read the CSV written by generate_dataset.py into a dataframe."""
    dataframe = pd.read_csv(dataset_file)
    print(f"Loaded {dataset_file} : {len(dataframe)} rows, "
          f"{dataframe['snr_db'].nunique()} SNR values")
    return dataframe


def get_tap_arrays(dataframe, num_taps=NUM_TAPS):
    """
    Pull the per-tap quantities out of the dataframe as plain NumPy matrices.

    Every row of the CSV is one channel realisation described by
    h_real_0..h_real_5, h_imag_0..h_imag_5 and tau_0..tau_5.

    Returns
    -------
    tap_amplitudes : (n_rows, num_taps) float
        |h_l| : magnitude of each complex tap.
    tap_powers : (n_rows, num_taps) float
        |h_l|^2 : instantaneous power of each tap.
    tap_delays : (n_rows, num_taps) float
        tau_l in nanoseconds (already relative, tau_0 = 0).
    """
    real_parts = dataframe[[f"h_real_{tap}" for tap in range(num_taps)]].to_numpy()
    imag_parts = dataframe[[f"h_imag_{tap}" for tap in range(num_taps)]].to_numpy()
    tap_delays = dataframe[[f"tau_{tap}" for tap in range(num_taps)]].to_numpy()

    tap_amplitudes = np.sqrt(real_parts ** 2 + imag_parts ** 2)   # |h_l|
    tap_powers     = tap_amplitudes ** 2                          # |h_l|^2

    return tap_amplitudes, tap_powers, tap_delays


def compute_kurtosis(tap_amplitudes):
    """
    Kurtosis of the received power, Eq. (2) of the reference paper:

        K = E[(|h| - mu)^4] / sigma^4

    The moments are taken over the L taps of a single snapshot, so the mean
    and standard deviation are population (not sample) statistics.

    Physically: a LOS snapshot has one dominant tap sitting far out in the
    tail of the amplitude distribution, which makes the distribution peaked
    and pushes kurtosis up. NLOS amplitudes are comparatively flat.
    """
    mean_amplitude = tap_amplitudes.mean(axis=1, keepdims=True)
    std_amplitude  = tap_amplitudes.std(axis=1, keepdims=True)   # ddof = 0

    deviations     = tap_amplitudes - mean_amplitude
    fourth_moment  = (deviations ** 4).mean(axis=1)

    # np.finfo(float).eps guards the (impossible in practice) zero-variance case
    return fourth_moment / (std_amplitude.ravel() ** 4 + np.finfo(float).eps)


def compute_skewness(tap_amplitudes):
    """
    Skewness of the received power, Eq. (5) of the reference paper:

        S = E[(|h| - mu)^3] / sigma^3

    A Rayleigh (NLOS) amplitude distribution is more skewed than a Rician
    (LOS) one, so this feature carries LOS/NLOS information as well.
    """
    mean_amplitude = tap_amplitudes.mean(axis=1, keepdims=True)
    std_amplitude  = tap_amplitudes.std(axis=1, keepdims=True)

    deviations     = tap_amplitudes - mean_amplitude
    third_moment   = (deviations ** 3).mean(axis=1)

    return third_moment / (std_amplitude.ravel() ** 3 + np.finfo(float).eps)


def compute_rising_time(tap_amplitudes, tap_delays):
    """
    Rising time, Eq. (6) of the reference paper:

        delta_tau = tau(argmax |h|) - min(tau)

    i.e. the delay gap between the first arriving path and the strongest
    path. In LOS the first path *is* the strongest, so the rising time
    collapses to ~0 ns. In NLOS the first path is attenuated by blockage or
    diffraction and the strongest path arrives later, giving a larger value.
    """
    strongest_tap_index = np.argmax(tap_amplitudes, axis=1)
    row_index           = np.arange(tap_amplitudes.shape[0])

    delay_of_strongest  = tap_delays[row_index, strongest_tap_index]
    delay_of_first      = tap_delays.min(axis=1)

    return delay_of_strongest - delay_of_first


def compute_rms_delay_spread(tap_powers, tap_delays):
    """
    RMS delay spread, Eqs. (7)-(8) of the reference paper.

    Mean excess delay (power-weighted first moment of the PDP):
        tau_m   = sum(tau_l * |h_l|^2) / sum(|h_l|^2)

    RMS delay spread (power-weighted second central moment):
        tau_rms = sqrt( sum((tau_l - tau_m)^2 * |h_l|^2) / sum(|h_l|^2) )

    NLOS power is spread across many comparable taps with long delays, so
    tau_rms is large; LOS power is concentrated on the first tap, so it is
    small. Units are nanoseconds, inherited from the tau columns.
    """
    total_power = tap_powers.sum(axis=1, keepdims=True) + np.finfo(float).eps

    mean_excess_delay = (tap_delays * tap_powers).sum(axis=1, keepdims=True) / total_power

    squared_spread = (((tap_delays - mean_excess_delay) ** 2) * tap_powers).sum(axis=1) \
                     / total_power.ravel()

    return np.sqrt(squared_spread)


def compute_rician_k_factor(tap_amplitudes):
    """
    Rician K-factor approximation, Eq. (9) of the reference paper:

        K_r = |h|_max^2 / (2 * sigma_|h|^2)

    Ratio of the dominant component's power to the variance of the tap
    amplitudes. Note this is the *linear* K-factor, not dB - the paper's
    Eq. (9) is used verbatim so the feature matches the reference.
    """
    peak_amplitude     = tap_amplitudes.max(axis=1)
    amplitude_variance = tap_amplitudes.var(axis=1)               # sigma^2

    return (peak_amplitude ** 2) / (2 * amplitude_variance + np.finfo(float).eps)


def add_features_to_dataframe(dataframe):
    """
    Part (a) entry point: compute all five features for every row and append
    them as new columns. Returns the augmented dataframe.
    """
    tap_amplitudes, tap_powers, tap_delays = get_tap_arrays(dataframe)

    dataframe["kurtosis"]         = compute_kurtosis(tap_amplitudes)
    dataframe["skewness"]         = compute_skewness(tap_amplitudes)
    dataframe["rising_time"]      = compute_rising_time(tap_amplitudes, tap_delays)
    dataframe["rms_delay_spread"] = compute_rms_delay_spread(tap_powers, tap_delays)
    dataframe["rician_k_factor"]  = compute_rician_k_factor(tap_amplitudes)

    print("\nPart (a): added feature columns ->", FEATURE_NAMES)
    return dataframe


def print_feature_summary(dataframe):
    """
    Sanity check for Part (a): show the class-conditional mean of each
    feature at the highest SNR, where the features are least noise-corrupted.
    The LOS/NLOS ordering of every mean should match the physical reasoning
    written in the docstrings above.
    """
    highest_snr = dataframe["snr_db"].max()
    subset      = dataframe[dataframe["snr_db"] == highest_snr]

    summary = subset.groupby("label")[FEATURE_NAMES].mean().T
    summary.columns = ["NLOS (label = -1)", "LOS (label = +1)"]

    print(f"\nClass-conditional feature means at SNR = {highest_snr} dB:")
    print(summary.to_string(float_format=lambda value: f"{value:10.4f}"))


# =============================================================================
# SPLITTING HELPER (shared by Parts (b) and (c))
# =============================================================================

def build_per_snr_splits(dataframe):
    """
    Create one stratified 80/20 train-test split per SNR value.

    The same splits are reused by Part (b) and Part (c) so that the two
    curves in the Part (c) figure are compared on identical test sets and
    any difference between them is caused only by the training strategy.

    Returns
    -------
    dict : snr_db -> (X_train, X_test, y_train, y_test)
           X arrays hold all five features in FEATURE_NAMES order.
    """
    splits = {}

    for snr_db in sorted(dataframe["snr_db"].unique()):
        snr_subset = dataframe[dataframe["snr_db"] == snr_db]

        feature_matrix = snr_subset[FEATURE_NAMES].to_numpy()
        labels         = snr_subset["label"].to_numpy()

        # stratify keeps the LOS/NLOS balance identical in train and test
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            labels,
            test_size=TEST_FRACTION,
            random_state=SPLIT_RANDOM_SEED,
            stratify=labels,
        )
        splits[snr_db] = (X_train, X_test, y_train, y_test)

    return splits


def train_and_score_svm(X_train, y_train, X_test, y_test):
    """
    Standard-scale, fit an RBF SVM with C = 1, and return test accuracy in %.

    The scaler is fitted on the training set only and then applied to the
    test set, as required by the assignment - fitting it on the full data
    would leak test-set statistics into training.
    """
    scaler          = StandardScaler().fit(X_train)
    X_train_scaled  = scaler.transform(X_train)
    X_test_scaled   = scaler.transform(X_test)

    classifier = SVC(kernel=SVM_KERNEL, C=SVM_C)
    classifier.fit(X_train_scaled, y_train)

    predictions = classifier.predict(X_test_scaled)
    return 100.0 * accuracy_score(y_test, predictions)


# =============================================================================
# PART (b) : SIX CLASSIFIERS, ACCURACY VS SNR
# =============================================================================

def run_part_b(splits):
    """
    Train the six classifiers at every SNR and return a results table.

    SVM-1..SVM-5 each see a single feature (a one-column feature matrix);
    SVM-6 sees all five columns at once. Every classifier is retrained from
    scratch at every SNR ("train = test SNR" strategy).

    Returns
    -------
    pandas.DataFrame indexed by snr_db, one column per classifier, values
    are test accuracies in percent.
    """
    accuracy_records = {}

    for snr_db, (X_train, X_test, y_train, y_test) in splits.items():
        row = {}

        # --- SVM-1 to SVM-5: one feature each ---------------------------------
        for feature_position, feature_name in enumerate(FEATURE_NAMES):
            # slice keeps the array 2-D, which is what scikit-learn expects
            single_feature_train = X_train[:, [feature_position]]
            single_feature_test  = X_test[:,  [feature_position]]

            row[FEATURE_LABELS[feature_name]] = train_and_score_svm(
                single_feature_train, y_train, single_feature_test, y_test
            )

        # --- SVM-6: all five features together --------------------------------
        row["SVM-6: All five features"] = train_and_score_svm(
            X_train, y_train, X_test, y_test
        )

        accuracy_records[snr_db] = row

    results = pd.DataFrame(accuracy_records).T
    results.index.name = "snr_db"

    print("\nPart (b): classification accuracy (%) vs SNR")
    print(results.to_string(float_format=lambda value: f"{value:7.2f}"))

    return results


def plot_part_b(results, output_file=None):
    """Single figure with all six accuracy-vs-SNR curves."""
    output_file = output_file or FIGURE_DIR / "partb_accuracy_vs_snr.png"

    figure, axis = plt.subplots(figsize=(9, 6))

    # The combined classifier is drawn thicker/black so it stands out.
    for column_name in results.columns:
        is_combined = column_name.startswith("SVM-6")
        axis.plot(
            results.index,
            results[column_name],
            marker="o",
            linewidth=2.6 if is_combined else 1.6,
            color="black" if is_combined else None,
            label=column_name,
        )

    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Classification accuracy (%)")
    axis.set_title("Part (b): LOS/NLOS classification accuracy vs SNR\n"
                   "(RBF-kernel SVM, C = 1, trained and tested at the same SNR)")
    axis.set_xticks(results.index)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    print(f"Saved figure: {output_file}")


# =============================================================================
# PART (c) : FIXED 25 dB MODEL VS MATCHED-SNR MODEL
# =============================================================================

def run_part_c(splits, part_b_results):
    """
    Train SVM-6 once on the SNR = 25 dB training split and evaluate that one
    frozen model (scaler included) on the test split of every SNR.

    The scaler is deliberately frozen too: in deployment you would not have
    the target-SNR statistics available to re-normalise with, so refitting it
    per SNR would quietly leak information the real system cannot have.

    Returns
    -------
    pandas.DataFrame indexed by snr_db with the two curves to be plotted.
    """
    X_train_25, _, y_train_25, _ = splits[FIXED_TRAIN_SNR]

    # --- fit scaler + SVM once, on 25 dB data only ---------------------------
    fixed_scaler = StandardScaler().fit(X_train_25)
    fixed_svm    = SVC(kernel=SVM_KERNEL, C=SVM_C)
    fixed_svm.fit(fixed_scaler.transform(X_train_25), y_train_25)

    # --- evaluate the frozen model at every SNR ------------------------------
    fixed_model_accuracy = {}
    for snr_db, (_, X_test, _, y_test) in splits.items():
        predictions = fixed_svm.predict(fixed_scaler.transform(X_test))
        fixed_model_accuracy[snr_db] = 100.0 * accuracy_score(y_test, predictions)

    results = pd.DataFrame({
        "Train = Test SNR": part_b_results["SVM-6: All five features"],
        f"Train at {FIXED_TRAIN_SNR} dB": pd.Series(fixed_model_accuracy),
    })
    results.index.name = "snr_db"
    results["Difference (matched - fixed)"] = (
        results["Train = Test SNR"] - results[f"Train at {FIXED_TRAIN_SNR} dB"]
    )

    print("\nPart (c): accuracy (%) of matched-SNR vs fixed 25 dB training")
    print(results.to_string(float_format=lambda value: f"{value:7.2f}"))

    return results


def plot_part_c(results, output_file=None):
    """Two-curve figure comparing the matched and fixed training strategies."""
    output_file = output_file or FIGURE_DIR / "partc_train25_vs_matched.png"

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.plot(results.index, results["Train = Test SNR"],
              marker="o", linewidth=2.4, color="tab:blue",
              label="Train = Test SNR (retrained per SNR)")

    axis.plot(results.index, results[f"Train at {FIXED_TRAIN_SNR} dB"],
              marker="s", linewidth=2.4, color="tab:red", linestyle="--",
              label=f"Train at {FIXED_TRAIN_SNR} dB (single fixed model)")

    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Classification accuracy (%)")
    axis.set_title("Part (c): matched-SNR training vs fixed 25 dB training\n"
                   "(SVM-6, all five features, RBF kernel, C = 1)")
    axis.set_xticks(results.index)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right", fontsize=10)

    figure.tight_layout()
    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    print(f"Saved figure: {output_file}")


# =============================================================================
# APPENDIX : DIAGNOSTIC FOR THE PART (b) DISCUSSION
# =============================================================================

def run_k_factor_definition_check(dataframe):
    """
    Diagnostic used to support the written discussion of the Rician K-factor.

    Eq. (9) of the paper approximates the K-factor as

        K_r = |h|_max^2 / (2 * sigma_|h|^2)

    where sigma_|h|^2 is the variance of the tap *amplitudes*. With only
    L = 6 taps and a single dominant tap, that dominant tap inflates the
    variance in the denominator as much as it inflates the numerator, so the
    ratio saturates near L^2 / (2(L-1)) = 3.6 no matter how strong the LOS
    component actually is. The feature therefore loses most of its
    discriminating power.

    The textbook definition instead compares the dominant tap against the
    *residual* power,

        K_alt = P_peak / (P_total - P_peak),

    which does not suffer from that cancellation. This function trains
    single-feature SVMs on both definitions so the difference can be quoted
    in the report. It does not affect Parts (a)-(c), which use Eq. (9) as
    specified in the assignment.
    """
    tap_amplitudes, tap_powers, _ = get_tap_arrays(dataframe)

    peak_power     = tap_powers.max(axis=1)
    residual_power = tap_powers.sum(axis=1) - peak_power
    alternative_k  = peak_power / (residual_power + np.finfo(float).eps)

    comparison = {}
    for snr_db in sorted(dataframe["snr_db"].unique()):
        snr_mask = (dataframe["snr_db"] == snr_db).to_numpy()
        labels   = dataframe.loc[snr_mask, "label"].to_numpy()

        row = {}
        for definition_name, feature_values in [
            ("Eq. (9) K-factor",  dataframe.loc[snr_mask, "rician_k_factor"].to_numpy()),
            ("P_peak / P_residual", alternative_k[snr_mask]),
        ]:
            feature_column = feature_values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(
                feature_column, labels,
                test_size=TEST_FRACTION,
                random_state=SPLIT_RANDOM_SEED,
                stratify=labels,
            )
            row[definition_name] = train_and_score_svm(X_train, y_train, X_test, y_test)

        comparison[snr_db] = row

    results = pd.DataFrame(comparison).T
    results.index.name = "snr_db"

    print("\nAppendix: accuracy (%) of the two Rician K-factor definitions")
    print(results.to_string(float_format=lambda value: f"{value:7.2f}"))

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ---- Part (a) -----------------------------------------------------------
    dataframe = load_raw_dataset()
    dataframe = add_features_to_dataframe(dataframe)
    print_feature_summary(dataframe)
    dataframe.to_csv(DATA_DIR / "los_nlos_dataset_with_features.csv", index=False)
    print("Saved dataset with features: los_nlos_dataset_with_features.csv")

    # ---- Shared splits ------------------------------------------------------
    splits = build_per_snr_splits(dataframe)

    # ---- Part (b) -----------------------------------------------------------
    part_b_results = run_part_b(splits)
    part_b_results.to_csv(RESULTS_DIR / "partb_accuracy_table.csv")
    plot_part_b(part_b_results)

    # ---- Part (c) -----------------------------------------------------------
    part_c_results = run_part_c(splits, part_b_results)
    part_c_results.to_csv(RESULTS_DIR / "partc_accuracy_table.csv")
    plot_part_c(part_c_results)

    # ---- Appendix diagnostic (supports the Part (b) discussion) -------------
    k_factor_check = run_k_factor_definition_check(dataframe)
    k_factor_check.to_csv(RESULTS_DIR / "appendix_kfactor_definitions.csv")


if __name__ == "__main__":
    main()
