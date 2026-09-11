"""
Dataset Generation for LOS/NLOS Channel Classification
COM 837 - ML for Wireless Communication Systems

Generates synthetic LOS (Rician) and NLOS (Rayleigh) channel realisations
at multiple SNR values and saves everything to a single CSV file.

CSV columns:
    snr_db          - SNR value in dB (used to filter data per SNR)
    label           - +1 for LOS, -1 for NLOS
    h_real_0 to 5  - real part of each noisy tap amplitude
    h_imag_0 to 5  - imaginary part of each noisy tap amplitude
    tau_0 to 5     - relative tap delays in nanoseconds (tau_0 = 0 always)
"""

import numpy as np
import pandas as pd

# Parameters
N_SAMPLES       = 1000                           # samples per class per SNR
L               = 6                             # number of multipath taps
K_DB            = 9.0                           # Rician K-factor (dB)

MEAN_DELAY_LOS  = 30.0                          # exponential delay mean AND power decay constant for LOS scattered paths (ns)
MEAN_DELAY_NLOS = 100.0                         # exponential delay mean AND power decay constant for NLOS paths (ns)
SNR_DB_VALUES   = [0, 5, 10, 15, 20, 25, 30]    # SNR values to generate (dB)

np.random.seed(67)

# Channel generation functions
def generate_LOS(n_samples):
    """
    Generate Rician (LOS) channel tap amplitudes and relative delays.

    Model
    -   ALL L tap delays are drawn from Exp(MEAN_DELAY_LOS) and sorted.
    -   Delays are then normalised by subtracting the minimum, so the
        earliest arriving path always has relative delay = 0 ns.
        This is the physically correct baseband representation.
    -   Tap 0 (earliest path) carries the dominant LOS component:
        amplitude = sqrt(K / (K+1)) * exp(j * phi),   phi ~ Uniform[0, 2*pi)
    -   Taps 1 to L-1 are scattered components that share the remaining
        power 1/(K+1) using an exponential decay profile:
        P_l propotional to exp( -tau_l / MEAN_DELAY_LOS )
        so later-arriving scattered paths are weaker than earlier ones.

    Returns
        channels : (n_samples, L) complex  - tap amplitudes
        delays   : (n_samples, L) float    - relative tap delays in ns
    """
    K = 10 ** (K_DB / 10)      # K-factor linear scale

    channels = np.zeros((n_samples, L), dtype=complex)
    delays   = np.zeros((n_samples, L))

    for i in range(n_samples):

        # --- Delays ---
        # Draw all L delays from exponential distribution and sort
        raw_delays  = np.random.exponential(scale=MEAN_DELAY_LOS, size=L)
        raw_delays  = np.sort(raw_delays)
        # Normalise: subtract minimum so earliest tap is at relative delay 0
        delays[i]   = raw_delays - raw_delays[0]

        # --- LOS tap (index 0) ---
        # Deterministic amplitude, random phase each realisation
        phi = np.random.uniform(0, 2 * np.pi)
        channels[i, 0] = np.sqrt(K / (K + 1)) * np.exp(1j * phi)

        # --- Scattered taps (indices 1 to L-1) ---
        # Exponential power decay: P_l propotional to exp(-tau_l / MEAN_DELAY_LOS)
        scatter_delays  = delays[i, 1:]
        decay_weights   = np.exp(-scatter_delays / MEAN_DELAY_LOS)

        # Normalise weights so total scattered power = 1 / (K + 1)
        scatter_powers  = (decay_weights / decay_weights.sum()) / (K + 1)

        # Complex Gaussian amplitudes: E[|h|^2] = P_l -> sigma = sqrt(P_l / 2)
        sigma = np.sqrt(scatter_powers / 2)
        channels[i, 1:] = (np.random.randn(L - 1) + 1j * np.random.randn(L - 1)) * sigma

    return channels, delays


def generate_NLOS(n_samples):
    """
    Generate Rayleigh (NLOS) channel tap amplitudes and relative delays.

    Model
    - All L tap delays are drawn from Exp(MEAN_DELAY_NLOS) and sorted.
    - Delays are normalised (subtract minimum) -> earliest tap at 0 ns.
    - All L taps are scattered (no dominant component, K = 0).
    - Tap powers follow an exponential decay profile:
      P_l propotional to exp( -tau_l / MEAN_DELAY_NLOS )
      Normalised so total power across all taps = 1.

    Returns
        channels : (n_samples, L) complex  - tap amplitudes
        delays   : (n_samples, L) float    - relative tap delays in ns
    """
    channels = np.zeros((n_samples, L), dtype=complex)
    delays   = np.zeros((n_samples, L))

    for i in range(n_samples):

        # --- Delays ---
        raw_delays  = np.random.exponential(scale=MEAN_DELAY_NLOS, size=L)
        raw_delays  = np.sort(raw_delays)
        delays[i]   = raw_delays - raw_delays[0]   # relative delays

        # --- Tap powers via exponential PDP ---
        decay_weights = np.exp(-delays[i] / MEAN_DELAY_NLOS)
        # Normalise so total power = 1
        tap_powers    = decay_weights / decay_weights.sum()

        # Complex Gaussian amplitudes
        sigma = np.sqrt(tap_powers / 2)
        channels[i]  = (np.random.randn(L) + 1j * np.random.randn(L)) * sigma

    return channels, delays


def add_noise(channels, snr_db):
    """
    Add complex AWGN to channel tap amplitudes at a given SNR.

    SNR definition:
        SNR = total channel power / total noise power
            = 1 / (L * noise_var_per_tap)    [since total channel power = 1]

    So:  noise_var_per_tap = 1 / (L * SNR_linear)

    Args:
        channels : (n_samples, L) complex array of clean tap amplitudes
        snr_db   : float, SNR in dB

    Returns:
        noisy_channels : (n_samples, L) complex array
    """
    snr_lin   = 10 ** (snr_db / 10)
    noise_var = 1.0 / (L * snr_lin)
    sigma_n   = np.sqrt(noise_var / 2)

    noise = (  np.random.randn(*channels.shape)
             + 1j * np.random.randn(*channels.shape)) * sigma_n

    return channels + noise


# Build and export dataset
def build_and_save_csv(filename='los_nlos_dataset.csv'):
    """
    Generate LOS and NLOS channel data at all SNR values,
    combine into a single DataFrame, and save as CSV.

    Each row = one channel realisation at one SNR value.

    How to use in feature extraction:
        df      = pd.read_csv('los_nlos_dataset.csv')
        snr_20  = df[df['snr_db'] == 20].reset_index(drop=True)
        # compute PDP from h_real/h_imag columns
        # extract K-factor, RMS delay spread etc. from PDP and tau columns
        # use 'label' column as the SVM target variable
    """

    print("Generating LOS channels ...")
    los_ch, los_delays = generate_LOS(N_SAMPLES)

    print("Generating NLOS channels ...")
    nlos_ch, nlos_delays = generate_NLOS(N_SAMPLES)

    all_rows = []

    for snr_db in SNR_DB_VALUES:
        print(f"  Adding noise at SNR = {snr_db} dB ...")

        los_noisy  = add_noise(los_ch,  snr_db)
        nlos_noisy = add_noise(nlos_ch, snr_db)

        # --- LOS rows (label = +1) ---
        for i in range(N_SAMPLES):
            row = {'snr_db': snr_db, 'label': 1}
            for tap in range(L):
                row[f'h_real_{tap}'] = los_noisy[i, tap].real
                row[f'h_imag_{tap}'] = los_noisy[i, tap].imag
                row[f'tau_{tap}']    = los_delays[i, tap]
            all_rows.append(row)

        # --- NLOS rows (label = -1) ---
        for i in range(N_SAMPLES):
            row = {'snr_db': snr_db, 'label': -1}
            for tap in range(L):
                row[f'h_real_{tap}'] = nlos_noisy[i, tap].real
                row[f'h_imag_{tap}'] = nlos_noisy[i, tap].imag
                row[f'tau_{tap}']    = nlos_delays[i, tap]
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(filename, index=False)

    print(f"\nSaved to {filename}")
    print(f"  Total rows   : {len(df)}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"  Rows per SNR : {len(df) // len(SNR_DB_VALUES)}  "
          f"({N_SAMPLES} LOS + {N_SAMPLES} NLOS)")

    return df


# Sanity checks

def sanity_check(df):
    """
    Verify the dataset looks physically correct.
    Checks power normalisation, delay structure, and K-factor separation.
    """
    print("\n--- Sanity Checks (clean channels, SNR = 30 dB) ---")

    snr_30 = df[df['snr_db'] == 30]
    los    = snr_30[snr_30['label'] ==  1]
    nlos   = snr_30[snr_30['label'] == -1]

    # 1. Power check - total channel power should be close to 1
    for name, subset in [('LOS', los), ('NLOS', nlos)]:
        pdp_cols = [f'h_real_{t}' for t in range(L)] + \
                   [f'h_imag_{t}' for t in range(L)]
        total_power = sum(
            subset[f'h_real_{t}']**2 + subset[f'h_imag_{t}']**2
            for t in range(L)
        )
        print(f"  Mean total power  {name} : {total_power.mean():.4f}  (expected ≈ 1.0)")

    # 2. tau_0 check - should be 0 for both classes after normalisation
    print(f"\n  Mean tau_0  LOS  : {los['tau_0'].mean():.6f} ns  (expected = 0.0)")
    print(f"  Mean tau_0  NLOS : {nlos['tau_0'].mean():.6f} ns  (expected = 0.0)")

    # 3. K-factor check - LOS should have much higher K than NLOS
    for name, subset in [('LOS', los), ('NLOS', nlos)]:
        pdp = np.array([
            subset[f'h_real_{t}'].values**2 + subset[f'h_imag_{t}'].values**2
            for t in range(L)
        ]).T                                        # shape (n_samples, L)
        peak        = pdp.max(axis=1)
        total       = pdp.sum(axis=1)
        k_est_db    = 10 * np.log10(peak / (total - peak + 1e-12))
        print(f"  Mean K-factor est {name} : {k_est_db.mean():.2f} dB")

    print(f"  (LOS target K = {K_DB} dB, NLOS target K ≈ 0 dB)\n")


# Run
df = build_and_save_csv('los_nlos_dataset.csv')

sanity_check(df)

print("Class balance check:")
for snr in SNR_DB_VALUES:
    subset = df[df['snr_db'] == snr]
    n_los  = (subset['label'] ==  1).sum()
    n_nlos = (subset['label'] == -1).sum()
    print(f"  SNR {snr:2d} dB  ->  LOS: {n_los},  NLOS: {n_nlos}")