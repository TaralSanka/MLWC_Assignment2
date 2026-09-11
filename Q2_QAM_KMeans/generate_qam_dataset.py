"""
=============================================================================
COM 837 - ML for Wireless Communication Systems
Assignment 2, Question 2 : Dataset generation
=============================================================================

Synthesises complex baseband 16-QAM realisations over an AWGN channel and
writes them to a single CSV file.

Specification followed (from the assignment):
    - Square 16-QAM grid, in-phase and quadrature coordinates from {-3,-1,1,3}
    - Constellation scaled so that the average symbol energy Es = 1
    - SNR values: [0, 5, 10, 15, 20, 25, 30] dB
    - Zero-mean complex AWGN added to the transmitted symbols
    - 200 samples per constellation point at every SNR value
    - Random seed = 67

CSV columns:
    snr_db      - SNR value in dB (filter on this to work at one SNR)
    symbol_id   - ground-truth constellation index, 0..15
    tx_I, tx_Q  - transmitted (noiseless) scaled constellation coordinates
    rx_I, rx_Q  - received noisy coordinates  = tx + AWGN
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
PAM_LEVELS         = np.array([-3.0, -1.0, 1.0, 3.0])   # per-axis grid levels
SAMPLES_PER_SYMBOL = 200                                # per constellation point, per SNR
SNR_DB_VALUES      = [0, 5, 10, 15, 20, 25, 30]         # SNR values to generate (dB)
RANDOM_SEED        = 67                                 # fixed by the assignment
DATA_DIR           = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE        = DATA_DIR / "qam16_dataset.csv"

np.random.seed(RANDOM_SEED)


# -----------------------------------------------------------------------------
# Constellation
# -----------------------------------------------------------------------------
def build_16qam_constellation():
    """
    Build the unit-average-energy square 16-QAM constellation.

    The raw grid {-3,-1,1,3} x {-3,-1,1,3} has average symbol energy

        E[|s|^2] = E[I^2] + E[Q^2] = 5 + 5 = 10

    so every point is divided by sqrt(10) to force Es = 1. Working at Es = 1
    means the noise variance below is exactly 1/SNR_linear, with no extra
    energy bookkeeping.

    Returns
    -------
    constellation : (16,) complex   - the scaled symbol alphabet
    symbol_ids    : (16,) int       - index 0..15 identifying each symbol
    """
    # meshgrid then ravel gives a deterministic, reproducible symbol ordering
    in_phase_grid, quadrature_grid = np.meshgrid(PAM_LEVELS, PAM_LEVELS, indexing="ij")
    raw_constellation = in_phase_grid.ravel() + 1j * quadrature_grid.ravel()

    average_energy = np.mean(np.abs(raw_constellation) ** 2)   # = 10.0
    constellation  = raw_constellation / np.sqrt(average_energy)

    symbol_ids = np.arange(constellation.size)

    # Guard: the normalisation must give exactly unit average energy
    assert np.isclose(np.mean(np.abs(constellation) ** 2), 1.0)

    return constellation, symbol_ids


def add_awgn(symbols, snr_db):
    """
    Add zero-mean complex AWGN at the requested SNR.

    With Es = 1, SNR = Es / N0 gives total noise power N0 = 1 / SNR_linear.
    That power is split equally between the I and Q dimensions, so each real
    dimension gets variance N0 / 2.

    Args:
        symbols : (n,) complex array of transmitted symbols
        snr_db  : float, SNR in dB

    Returns:
        (n,) complex array of received symbols
    """
    snr_linear           = 10 ** (snr_db / 10)
    total_noise_power    = 1.0 / snr_linear
    noise_std_per_axis   = np.sqrt(total_noise_power / 2)

    noise = (np.random.randn(symbols.size)
             + 1j * np.random.randn(symbols.size)) * noise_std_per_axis

    return symbols + noise


# -----------------------------------------------------------------------------
# Build and export
# -----------------------------------------------------------------------------
def build_and_save_csv(output_file=OUTPUT_FILE):
    """
    Generate SAMPLES_PER_SYMBOL noisy realisations of every constellation
    point at every SNR, and write one flat CSV.
    """
    constellation, symbol_ids = build_16qam_constellation()

    # Transmitted symbol sequence for one SNR: each symbol repeated 200 times.
    transmitted_symbols = np.repeat(constellation, SAMPLES_PER_SYMBOL)
    transmitted_ids     = np.repeat(symbol_ids,    SAMPLES_PER_SYMBOL)

    all_blocks = []
    for snr_db in SNR_DB_VALUES:
        print(f"  Generating SNR = {snr_db} dB ...")
        received_symbols = add_awgn(transmitted_symbols, snr_db)

        all_blocks.append(pd.DataFrame({
            "snr_db":    snr_db,
            "symbol_id": transmitted_ids,
            "tx_I":      transmitted_symbols.real,
            "tx_Q":      transmitted_symbols.imag,
            "rx_I":      received_symbols.real,
            "rx_Q":      received_symbols.imag,
        }))

    dataframe = pd.concat(all_blocks, ignore_index=True)
    dataframe.to_csv(output_file, index=False)

    print(f"\nSaved to {output_file}")
    print(f"  Total rows       : {len(dataframe)}")
    print(f"  Rows per SNR     : {len(dataframe) // len(SNR_DB_VALUES)} "
          f"(16 symbols x {SAMPLES_PER_SYMBOL} samples)")
    print(f"  Columns          : {list(dataframe.columns)}")

    return dataframe


def sanity_check(dataframe):
    """
    Verify the generated dataset is physically consistent:
      1. transmitted constellation has unit average energy
      2. every symbol has exactly SAMPLES_PER_SYMBOL realisations per SNR
      3. measured received SNR matches the requested SNR
    """
    print("\n--- Sanity checks ---")

    transmitted_energy = (dataframe["tx_I"] ** 2 + dataframe["tx_Q"] ** 2).mean()
    print(f"  Mean transmitted symbol energy : {transmitted_energy:.4f}  (expected 1.0)")

    counts_per_symbol = dataframe.groupby(["snr_db", "symbol_id"]).size().unique()
    print(f"  Samples per symbol per SNR     : {counts_per_symbol}  "
          f"(expected [{SAMPLES_PER_SYMBOL}])")

    print("\n  Requested vs measured SNR:")
    for snr_db in SNR_DB_VALUES:
        subset          = dataframe[dataframe["snr_db"] == snr_db]
        noise_i         = subset["rx_I"] - subset["tx_I"]
        noise_q         = subset["rx_Q"] - subset["tx_Q"]
        measured_noise  = (noise_i ** 2 + noise_q ** 2).mean()
        measured_snr_db = 10 * np.log10(1.0 / measured_noise)
        print(f"    requested {snr_db:2d} dB  ->  measured {measured_snr_db:6.2f} dB")


if __name__ == "__main__":
    dataset = build_and_save_csv()
    sanity_check(dataset)
