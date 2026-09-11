# Question 2 — Constellation Demodulation & Clustering using K-means

**Setup.** `generate_qam_dataset.py` synthesises a square 16-QAM constellation with I/Q
coordinates from {−3, −1, 1, 3}, scaled by 1/√10 so the average symbol energy is exactly 1.
Zero-mean complex AWGN of total power 1/SNR (split evenly across I and Q) is added at
SNR ∈ {0, 5, 10, 15, 20, 25, 30} dB, with 200 samples per constellation point per SNR and
`np.random.seed(67)`. That gives 3200 rows per SNR and 22,400 rows overall. The sanity check
confirms unit transmit energy, exactly 200 samples per symbol per SNR, and measured SNRs
within 0.1 dB of the requested values.

All K-means fits use `k-means++`, `n_init = 10`, `random_state = 42`.

Useful reference number throughout: after scaling, the **nearest-neighbour symbol spacing is
2/√10 = 0.6325**, and the per-axis noise standard deviation is √(1/(2·SNR_lin)).

| SNR (dB) | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| σ per axis | 0.707 | 0.398 | 0.224 | 0.126 | 0.071 | 0.040 | 0.022 |
| σ / symbol spacing | **1.12** | 0.63 | 0.35 | 0.20 | 0.11 | 0.06 | 0.04 |

At 0 dB the noise standard deviation is *larger than the distance between adjacent symbols*.
That single fact explains most of what follows.

---

## Part (a) — Polar feature extraction

Two features are appended per received sample:

- **Instantaneous amplitude** r = √(rx_I² + rx_Q²), Eq. (1)
- **Instantaneous phase** θ, Eq. (2), in radians

**Implementation note on the phase.** Eq. (2) is written as arctan(rx_Q / rx_I), but
`np.arctan2(rx_Q, rx_I)` is used. Plain `arctan` returns values in (−π/2, π/2) only, so it
maps a point and its negation to the *same* phase: the symbol at (+3, +3) and the symbol at
(−3, −3) become indistinguishable, and the constellation folds onto half the plane. This is
not a hypothetical — the appendix diagnostic re-runs Feature Set 2 with both definitions at
25 dB:

| Phase definition | Feature Set 2 purity |
|---|---|
| arctan2(Q, I) — used here | **1.0000** |
| arctan(Q / I) — literal | 0.5188 |

The literal form loses almost exactly half the symbols, which is the expected signature of
quadrant folding (8 aliased pairs out of 16 symbols). arctan2 is the standard definition of
instantaneous phase and is used everywhere below.

---

## Part (b) — Model selection, clustering, and feature-set comparison

### b1–b2: Inertia and silhouette vs K at SNR = 25 dB

**Figure:** `partb_elbow_silhouette.png`

| K | 12 | 13 | 14 | 15 | **16** | 17 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|
| Inertia | 167.6 | 129.0 | 89.6 | 49.4 | **10.2** | 9.97 | 9.72 | 9.24 |
| Silhouette | 0.660 | 0.711 | 0.769 | 0.823 | **0.880** | 0.847 | 0.813 | 0.745 |

Both metrics identify K = 16 unambiguously, and they do it more sharply than a typical
clustering problem does — because unlike most real datasets, this one genuinely has 16
well-separated generative modes.

**Inertia** falls steeply and almost linearly from K = 12 to K = 15, then drops off a cliff
at K = 16 (49.4 → 10.2, a 79% reduction in one step) and goes essentially flat afterwards
(10.2 → 9.97 → 9.72). The interpretation is direct: below K = 16 at least one cluster is
forced to straddle two distinct symbol blobs, and the WCSS is dominated by the inter-symbol
distance of that straddled pair. The moment the sixteenth centroid becomes available, every
cluster collapses onto exactly one blob, and the residual inertia is nothing but the AWGN
variance inside the blobs. Adding a seventeenth centroid can only split one noise cloud in
half, which buys almost nothing. This is a textbook elbow, and it sits at the true symbol
count.

**Silhouette** peaks at exactly K = 16 (0.880) and declines on both sides. The decline above
16 is the informative half: splitting a single isotropic Gaussian blob produces two clusters
whose members are nearly as close to the neighbouring cluster as to their own, which drives
the silhouette down. Below 16 there are two smaller local bumps, at K = 4 (0.511) and K = 8
(0.546) — the constellation's own coarse structure, corresponding to the four quadrants and
to the four rows/eight half-rows of the grid. K-means finds those as legitimate but coarser
partitions before it resolves individual symbols.

### b3: K = 16 clustering at SNR = 25 dB

**Figure:** `partb_clusters_k16_25db.png`

The 16 learned centroids land on the 16 true constellation points, with an RMS displacement
of 0.0039 against a symbol spacing of 0.6325 — an error of 0.6%. Every noise cloud is
visibly compact and separated by clear empty space. **Cluster purity = 1.0000**: at 25 dB,
completely unsupervised K-means recovers the 16-QAM alphabet perfectly, without ever being
told what a constellation is.

### b4: Cluster purity across the three feature sets

At the required SNR of 25 dB, all three feature sets are **saturated**:

| Feature set | Purity at 25 dB |
|---|---|
| Set 1: (rx_I, rx_Q) | 1.0000 |
| Set 2: (r, θ) | 1.0000 |
| Set 3: (rx_I, rx_Q, r, θ), StandardScaler applied | 1.0000 |

This is a real result, not a failure to compute: at 25 dB the clusters are separated by ~16
noise standard deviations, so even a distorted metric has enough margin to get all 3200
samples right. But it cannot rank the three representations, which is what the question is
really after. Repeating the comparison at every SNR breaks the tie
(`partb_purity_by_featureset_all_snr.csv`):

| SNR (dB) | Set 1 (Cartesian) | Set 2 (Polar) | Set 3 (Both, scaled) |
|---|---|---|---|
| 0 | **0.2547** | 0.2469 | 0.2431 |
| 5 | **0.4491** | 0.4150 | 0.4113 |
| 10 | **0.7656** | 0.6966 | 0.6628 |
| 15 | **0.9841** | 0.9700 | 0.9678 |
| 20 | **0.9997** | 0.9994 | 0.9997 |
| 25 | 1.0000 | 1.0000 | 1.0000 |
| 30 | 1.0000 | 1.0000 | 1.0000 |

**Feature Set 1, the raw Cartesian pair, gives the highest purity** — it wins at every SNR
where the metric is not saturated, by up to 10 points (0.766 vs 0.663 at 10 dB). Set 2 is
second and Set 3 is last.

Set 3 losing is the mildly counter-intuitive part and is worth stating explicitly: adding
features does not add *information* here, because r and θ are a deterministic invertible
function of (rx_I, rx_Q). What Set 3 adds is **weight**. After `StandardScaler`, all four
axes carry equal variance, so the two distorted polar axes get exactly as much say in the
distance metric as the two well-conditioned Cartesian ones — the distortion described below
is imported into an otherwise clean representation and dilutes it. Redundant features are
not free when the algorithm's only knob is a distance.

### Why Euclidean distance suits Cartesian coordinates but distorts polar ones

**Cartesian is the right space because the noise lives there.** The channel adds i.i.d.
zero-mean Gaussian noise of equal variance to rx_I and rx_Q independently. Its density
contours are therefore *circles* in the I/Q plane, and the maximum-likelihood detector for
AWGN is the minimum-Euclidean-distance detector, whose decision regions are the Voronoi
cells of the constellation.

K-means is built on exactly the same two assumptions — isotropic clusters of equal spread,
partitioned by Euclidean Voronoi boundaries — so the algorithm's inductive bias is a precise
match for the physics of the channel. There is no modelling mismatch to pay for. On top of
that, the 16-QAM grid is uniformly spaced along both axes, so equal Euclidean distance
corresponds to equal error probability everywhere in the plane. Unsupervised K-means at K = 16
is, in effect, blindly rediscovering the ML detector.

**Polar coordinates break this in four separate ways.**

1. **The units do not commute.** r spans roughly [0, 1.34] while θ spans [−π, π], a range
   4.7× larger. In an unweighted Euclidean sum (Δr)² + (Δθ)², the phase axis silently
   dominates, so the algorithm partitions mostly by angle and under-uses amplitude. Adding
   a squared amplitude in volts to a squared angle in radians is not a physically meaningful
   quantity in the first place.

2. **Δθ is not proportional to real distance.** Two points at radius r separated by Δθ are
   physically 2r·sin(Δθ/2) apart, but polar Euclidean charges them Δθ regardless of r. Phase
   differences are therefore **over-weighted near the origin and under-weighted far from it**.
   For 16-QAM this is severe: the four inner symbols sit at r = 0.447 and the four corner
   symbols at r = 1.342, a 3× ratio. The inner symbols get their angular separation
   exaggerated threefold relative to the corners, so the metric is inconsistent across the
   very constellation it is meant to partition.

3. **θ is circular and the metric is not.** θ = +π and θ = −π are the same physical
   direction but are maximally far apart in Euclidean terms. Any cluster that straddles the
   negative real axis is torn in two. At 25 dB no symbol sits close enough to the wrap for
   this to bite, but at low SNR the noise pushes samples across it and the wrap becomes a
   real source of error.

4. **Neither polar coordinate alone separates the alphabet.** 16-QAM has only **three
   distinct radii** (0.447 for 4 symbols, 1.0 for 8, 1.342 for 4), so amplitude alone can
   never resolve 16 classes. And several symbols share a phase exactly — (1,1) and (3,3) are
   both at θ = π/4 — so phase alone cannot either. The two features must be combined with
   the *correct relative weighting*, and unweighted Euclidean is not it.

The net effect is decision boundaries that are wedges and annuli in the physical plane
rather than the square Voronoi cells the AWGN channel calls for. Set 2 still reaches purity
1.0 at 25 dB only because the clusters are so far apart that even a badly shaped boundary
separates them; the moment the noise grows, Set 2 falls behind Set 1 and stays behind.

---

## Part (c) — Adaptive K-means vs fixed 25 dB template

**Figure:** `partc_adaptive_vs_fixed.png`

| SNR (dB) | Adaptive K-Means | Fixed Template | Fixed − Adaptive |
|---|---|---|---|
| 0 | 0.2547 | **0.2631** | +0.0084 |
| 5 | 0.4491 | **0.4722** | **+0.0231** |
| 10 | 0.7656 | **0.7725** | +0.0069 |
| 15 | **0.9841** | 0.9838 | −0.0003 |
| 20 | 0.9997 | 0.9997 | 0.0000 |
| 25 | 1.0000 | 1.0000 | 0.0000 |
| 30 | 1.0000 | 1.0000 | 0.0000 |

### At high SNR, do the two curves agree?

Yes — identically. Both reach purity 1.0000 at 25 and 30 dB, and 0.9997 at 20 dB (a single
misassigned sample out of 3200). The curves are indistinguishable above ~15 dB.

This says the centroids are **fully identifiable and stable in the low-noise regime**. The
supplementary geometry table (`partc_centroid_geometry.csv`) shows how tightly:

| SNR (dB) | Adaptive centroid RMS error vs true grid | Adaptive min centroid gap |
|---|---|---|
| 0 | 0.5744 | 0.8561 |
| 5 | 0.2347 | 0.6555 |
| 10 | 0.0536 | 0.5680 |
| 15 | 0.0112 | 0.6046 |
| 20 | 0.0062 | 0.6177 |
| 25 | 0.0039 | 0.6258 |
| 30 | 0.0021 | 0.6285 |

*(True grid spacing = 0.6325; fixed template RMS error = 0.0039 by construction.)*

Above 15 dB the adaptive fit recovers the true constellation to within 1–2% of the symbol
spacing. Refitting converges to the same 16 points the template already holds, so it changes
nothing. Practically: once σ/spacing < 0.2, a receiver can learn the constellation once and
freeze it — per-SNR re-estimation is wasted computation.

### At low SNR, which strategy is more accurate?

**The fixed template**, consistently: +0.0231 at 5 dB, +0.0084 at 0 dB, +0.0069 at 10 dB. It
wins at every SNR below 15 dB.

The margin is modest in absolute terms, and that is itself worth being honest about: at 0 dB
both strategies are near-useless (0.25 and 0.26 against a 1/16 = 0.0625 chance floor), so the
template is not rescuing anything — the information has already been destroyed by the
channel. What the comparison shows is that the template never makes things *worse*, and the
adaptive fit sometimes does. The 5 dB point is where the difference peaks, because that is
where the clusters are still partly resolvable but no longer resolvable enough for K-means
to place its own centroids correctly.

### Geometric explanation: why fitting on low-SNR samples causes centroid merging

At 0 dB, σ per axis is 0.707 while adjacent symbols are 0.6325 apart, so **the noise standard
deviation exceeds the symbol spacing**. Each of the 16 Gaussian blobs has a radius larger
than the gap to its neighbour, and they superpose into one broad, roughly unimodal cloud. The
density minima that used to sit between symbols are gone — there is nothing left in the data
for a density-seeking algorithm to lock onto.

K-means does not care about this. It minimises within-cluster sum of squares, so given 3200
points drawn from one broad cloud and 16 centroids, it will do the only thing its objective
rewards: **tile the cloud into 16 roughly equal-mass Voronoi cells**. Those cells are placed
to balance the point mass, not to sit on the generative modes. The geometry table shows the
consequence at 0 dB: the adaptive centroids drift an RMS 0.5744 off the true grid — 91% of a
full symbol spacing, i.e. each centroid is nearly a whole symbol away from where it should
be — and their minimum pairwise gap swells to 0.8561, larger than the true grid's 0.6325.
They have spread outward to cover the noise tails, which occupy a much larger area than the
constellation itself.

The merging follows directly. If centroids migrate outward to tile the sparse but wide noise
tails, the crowded interior of the constellation is left under-served: several true symbols
end up sharing a single cluster, while other clusters cover regions containing no symbol
centre at all. Cluster purity's majority vote then credits only the most frequent symbol in
each merged cluster and counts the rest as errors — which is exactly the penalty this metric
is designed to apply.

The fixed template avoids all of this by **not being fitted to the data at all**. Its
centroids are pinned to the true constellation geometry, so `predict()` at any SNR reduces to
assigning each received sample to its nearest constellation point — a minimum-distance
detector, which is the ML detector for AWGN. It cannot recover information the noise has
destroyed, so its purity still falls to 0.26 at 0 dB. But it is *optimal given the noise*,
and it therefore acts as a floor that adaptive fitting can match at high SNR and can only
fall below at low SNR.

The general principle: **unsupervised clustering estimates geometry from data; a template
imposes geometry that is already known.** When the data are clean, the two agree and the
estimate is free. When the data are noisy, estimating a structure you already know is not
just wasteful — it actively injects error, because the estimator will faithfully fit the
noise you handed it.
