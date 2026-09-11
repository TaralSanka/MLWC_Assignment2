# Question 1 — SVM for LOS/NLOS Identification

**Setup.** `generate_dataset.py` was run unmodified (seed 67), producing 14,000 rows:
1000 LOS + 1000 NLOS realisations at each of SNR ∈ {0, 5, 10, 15, 20, 25, 30} dB, six taps
per realisation. All classifiers are RBF-kernel SVMs with C = 1, trained on a stratified
80/20 split (`random_state = 42`) with `StandardScaler` fitted on the training set only.
Parts (b) and (c) reuse the *same* splits so the two curves in Part (c) are compared on
identical test sets.

---

## Part (a) — Feature extraction

Five scalar features are computed per realisation from the six taps, following the
reference paper:

| Feature | Equation | Computed from |
|---|---|---|
| Kurtosis | Eq. (2) | 4th central moment of \|h_l\| / σ⁴ |
| Skewness | Eq. (5) | 3rd central moment of \|h_l\| / σ³ |
| Rising time | Eq. (6) | τ(argmax\|h_l\|) − min(τ_l) |
| RMS delay spread | Eqs. (7)–(8) | power-weighted 2nd central moment of the PDP |
| Rician K-factor | Eq. (9) | \|h\|²_max / (2σ²_\|h\|) |

Moments are population statistics over the L = 6 taps of a single snapshot (`ddof = 0`),
which is what the paper's per-snapshot definitions imply.

**Class-conditional means at SNR = 30 dB:**

| Feature | NLOS (−1) | LOS (+1) |
|---|---|---|
| Kurtosis | 2.16 | 3.87 |
| Skewness | 0.35 | 1.61 |
| Rising time (ns) | 28.15 | 0.00 |
| RMS delay spread (ns) | 45.19 | 6.98 |
| Rician K-factor (Eq. 9) | 6.56 | 4.58 |

Four of the five features order themselves exactly as the physics predicts. The LOS
snapshots are more peaked and more skewed (one dominant tap plus five weak scattered
taps), their strongest tap *is* their first tap so rising time is identically zero, and
their power is concentrated at short delay so the RMS delay spread is ~6.5× smaller than
NLOS.

The Eq. (9) K-factor is the exception — it comes out **lower** for LOS than for NLOS. This
is not a bug; see the discussion in Part (b).

---

## Part (b) — Six classifiers, accuracy vs SNR

**Figure:** `partb_accuracy_vs_snr.png`

| SNR (dB) | Kurtosis | Skewness | Rising time | RMS delay spread | Rician K | All five |
|---|---|---|---|---|---|---|
| 0 | 59.50 | 58.75 | 79.75 | 88.50 | 60.00 | **89.25** |
| 5 | 82.25 | 84.50 | 83.00 | 90.00 | 68.00 | **98.25** |
| 10 | 90.75 | 93.50 | 81.25 | 93.00 | 76.00 | **99.50** |
| 15 | 93.25 | 95.00 | 83.00 | 97.25 | 81.75 | **99.75** |
| 20 | 93.75 | 94.50 | 83.00 | 98.25 | 81.75 | **100.00** |
| 25 | 93.50 | 93.75 | 82.50 | 98.25 | 82.50 | **100.00** |
| 30 | 93.75 | 95.50 | 82.50 | 98.75 | 81.50 | **100.00** |

### Which single feature performs best overall, and why?

**RMS delay spread** (SVM-4). It is the best single feature at *every* SNR — 88.5% at 0 dB
rising to 98.75% at 30 dB — and it is the only one that stays above 88% across the whole
range.

The reason is baked into the channel model. The two classes are generated with different
delay statistics: LOS delays are drawn from Exp(30 ns) and its power decay constant is
30 ns, while NLOS uses Exp(100 ns) for both. So LOS and NLOS differ not only in *how* power
is distributed over the taps but in the *span of delays over which the taps are placed*.
τ_rms measures exactly that second central moment of the PDP, and it is fed by both effects
at once: the delay axis is stretched ~3.3× in NLOS, and the power weighting is spread
across all six taps instead of being concentrated on tap 0. The class-conditional means at
30 dB are 45.2 ns vs 7.0 ns — a separation of more than 6× with modest overlap.

Compare that with kurtosis, skewness and the K-factor, which use only the *amplitude*
distribution and throw the delay axis away. They estimate a shape statistic from six
samples, which is inherently high-variance, and they saturate around 93–95%.

### Does combining all five features help?

Yes, at every SNR, and the combined classifier beats the best single feature everywhere:

| SNR (dB) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| SVM-6 − best single | +0.75 | **+8.25** | **+6.50** | +2.50 | +1.75 | +1.75 | +1.25 |

**The improvement is most noticeable at 5 and 10 dB** (+8.25 and +6.50 points). That is the
transition region: the delay-domain feature is already partly informative there (90%), and
the amplitude-domain features have just become usable (kurtosis and skewness jump from ~59%
at 0 dB to 82–93% at 5–10 dB). Because the two families make *different* mistakes — one
looks at delay geometry, the other at amplitude statistics — their errors are only weakly
correlated and the RBF kernel can carve a boundary in 5-D that neither can find in 1-D.

At the two extremes the gain shrinks, for opposite reasons. At 0 dB the other four features
are close to noise (58–80%) and have almost nothing to contribute to τ_rms, so SVM-6 gains
only 0.75 points. At ≥ 20 dB τ_rms alone is already at 98%, so there is almost no headroom
left — although the combination does reach a clean 100%.

### Which feature fails at low SNR despite being physically meaningful?

**The Rician K-factor.** It is the single most physically direct statement of "is there a
dominant path?", and it is the feature that classical threshold-based LOS identification
has historically been built on — yet it is the worst performer at 0 dB (60.0%, barely above
chance) and the worst overall.

Two separate mechanisms are at work, and both are worth stating.

**(i) The low-SNR failure — noise floods the weak taps.** `add_noise` adds AWGN of variance
1/(L·SNR) per tap. At 0 dB that is 1/6 ≈ 0.167 per tap, against a *total* clean channel
power of 1. In a LOS realisation, taps 1–5 collectively carry only 1/(K+1) = 1/8.9 ≈ 0.11
of the power — i.e. roughly 0.02 per tap. The noise is therefore about **eight times
stronger than the scattered taps it is sitting on**. Those taps stop reporting the channel
and start reporting the noise floor, so the amplitude spread of a noisy LOS snapshot begins
to look like that of a noisy NLOS snapshot. The feature-drift table below shows this
directly: as SNR falls, the *LOS* statistics collapse towards the NLOS ones while the NLOS
statistics barely move, because NLOS taps are all of comparable size and are far less
distorted by an added noise floor.

| Feature (LOS mean → NLOS mean) | 30 dB | 10 dB | 0 dB |
|---|---|---|---|
| Kurtosis | 3.87 → 2.16 | 3.73 → 2.19 | **2.55 → 2.17** |
| Skewness | 1.61 → 0.35 | 1.53 → 0.39 | **0.69 → 0.30** |
| Rician K (Eq. 9) | 4.58 → 6.56 | 5.06 → 7.09 | **6.97 → 8.10** |
| RMS delay spread (ns) | 6.98 → 45.19 | 11.91 → 51.89 | 20.47 → 65.37 |

Kurtosis, skewness and the K-factor all converge to near-identical class means at 0 dB —
which is why all three sit at 58–60% accuracy there. τ_rms is the outlier that survives,
because its separation is anchored in the delay axis, and the τ columns are noiseless.

**(ii) An additional, SNR-independent problem with Eq. (9).** Eq. (9) normalises the peak
power by the variance of the tap *amplitudes*, σ²_|h|. With only L = 6 taps and one
dominant tap of amplitude a, that dominant tap inflates the denominator as much as the
numerator: σ²_|h| ≈ a²(L−1)/L², so K_r ≈ L²/(2(L−1)) = 3.6 **regardless of how strong the
LOS component actually is**. The estimator saturates. That is why the LOS mean (4.58) comes
out *below* the NLOS mean (6.56) and why the feature plateaus at ~82% even at 30 dB.

To confirm this is the definition and not the physics, the script also evaluates the
textbook form K = P_peak / (P_total − P_peak) — same feature, no cancellation
(`appendix_kfactor_definitions.csv`):

| SNR (dB) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| Eq. (9) | 60.00 | 68.00 | 76.00 | 81.75 | 81.75 | 82.50 | 81.50 |
| P_peak / P_residual | 65.75 | 87.75 | 96.25 | 96.75 | 97.50 | 98.00 | 98.00 |

Under the standard definition the K-factor becomes a near-best feature at high SNR (98%)
and still collapses at 0 dB (65.75%) — which is exactly the "physically meaningful but
fails at low SNR" behaviour the question is pointing at. Parts (a)–(c) use Eq. (9) as
specified; this check is reported only to separate the two causes.

---

## Part (c) — Fixed 25 dB model vs matched-SNR training

**Figure:** `partc_train25_vs_matched.png`

SVM-6 was trained once on the SNR = 25 dB training split (scaler frozen along with the
classifier, since a deployed receiver would not know the target-SNR statistics) and
evaluated on every SNR's test split.

| SNR (dB) | Train = Test SNR | Train at 25 dB | Gap |
|---|---|---|---|
| 0 | 89.25 | 58.00 | **−31.25** |
| 5 | 98.25 | 78.25 | **−20.00** |
| 10 | 99.50 | 97.50 | −2.00 |
| 15 | 99.75 | 99.50 | −0.25 |
| 20 | 100.00 | 99.75 | −0.25 |
| 25 | 100.00 | 100.00 | 0.00 |
| 30 | 100.00 | 100.00 | 0.00 |

### At high SNR, do the two curves agree?

Yes — they are identical at 25 and 30 dB and within 0.25 points at 15 and 20 dB. The two
curves are indistinguishable above ~12 dB.

This says the features have become **effectively SNR-invariant** once the noise floor drops
below the weakest taps. Looking at the drift table: between 15 and 30 dB the LOS kurtosis
moves only 3.83 → 3.87, skewness 1.59 → 1.61, rising time is pinned at 0.00, and τ_rms
moves 9.09 → 6.98 ns. The feature distributions have converged to their noiseless limits,
so the *same region of feature space* is occupied at 25 dB and at 30 dB. A decision
boundary fitted in that region is therefore valid at any other high SNR — there is nothing
left for retraining to adapt to. Practically: above ~15 dB you can train once and deploy,
and you lose nothing.

### At low SNR, which strategy wins, and by how much?

Matched training wins decisively: **+31.25 points at 0 dB** (89.25% vs 58.00%) and **+20.00
points at 5 dB** (98.25% vs 78.25%). At 0 dB the fixed model is 8 points off a coin flip;
it has essentially stopped working.

The reason is **covariate shift**. The 25 dB model has learned a boundary positioned for
tight, well-separated clusters — LOS around (kurtosis 3.87, τ_rms 7.2 ns), NLOS around
(2.16, 45.5 ns). At 0 dB the LOS cluster has migrated to (2.55, 20.5 ns) and the NLOS
cluster to (2.17, 65.4 ns). The LOS test points now land where the model's training data
put *NLOS*, so they are confidently misclassified. Two things compound this:

1. **The frozen `StandardScaler`.** It was fitted on 25 dB statistics, so 0 dB features are
   normalised by the wrong mean and standard deviation. Every feature arrives offset from
   where the RBF kernel expects it, and because the RBF kernel is *local*, points far from
   the training support fall into a region where the decision function has no meaningful
   support and defaults toward one class.
2. **The classes have genuinely moved closer together.** Even the matched model only reaches
   89.25% at 0 dB — the information is partly destroyed by the noise. A high-SNR model has
   never seen an overlapping pair of classes and has no notion of where to place a
   conservative boundary in that regime.

Generalising: a classifier trained at a single high SNR is implicitly a classifier of
*noiseless channel geometry*, and it assumes the features it is handed are estimates of
that geometry. At low SNR that assumption is violated — the features estimate the noise as
much as the channel.

### Would training at 0 dB and testing across all SNR be different?

**Yes, and asymmetrically so — but in the opposite direction, and it is not simply the
mirror image of the 25 dB case.**

Reasoning from the drift table: as SNR increases, noise-corrupted feature values move
*monotonically* toward their clean values, and crucially they move **inward** — the low-SNR
LOS and NLOS clusters are broad and overlapping, and the high-SNR clusters sit nested
*inside* that overlap region, at its two extremes. So a boundary drawn through the middle
of the smeared 0 dB distributions still lands between the two tight high-SNR clusters. It
will be in roughly the right place; it will just be badly *positioned* — placed to hedge
against noise that is no longer there, and therefore not tight enough to exploit the clean
separation. The 25 dB→0 dB direction has no such luck: the boundary sits at one extreme of
the space and the test points migrate right past it.

So the prediction is: a 0 dB-trained model should degrade gracefully rather than collapse,
holding roughly its own 0 dB accuracy (~89%) across the SNR range, but **never reaching the
100% that matched training achieves at high SNR** — it leaves ~10 points on the table where
the fixed 25 dB model leaves ~0.

Running it confirms this (diagnostic only, not part of the required deliverable):

| Trained at | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| 0 dB | 89.25 | 92.25 | 90.00 | 90.50 | 91.25 | 90.75 | 89.75 |
| 5 dB | 80.75 | 98.25 | 98.00 | 97.75 | 98.25 | 97.75 | 97.75 |
| 25 dB | 58.00 | 78.25 | 97.50 | 99.50 | 99.75 | 100.00 | 100.00 |

The 0 dB model is nearly flat at ~90% everywhere — far more *robust* than the 25 dB model
but with a hard ceiling. The practical takeaway is that **training at the worst SNR you
expect to see buys robustness at the cost of peak accuracy, and training at the best SNR
buys peak accuracy at the cost of catastrophic low-SNR failure**. The 5 dB row shows the
sensible compromise: ~98% from 5 dB upward and only an 8.5-point loss at 0 dB. If a single
fixed model must be deployed, it should be trained near the *low end* of the operating
range, not the high end — or better, trained on data pooled across SNRs.
