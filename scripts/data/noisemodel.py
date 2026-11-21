"""
noisemodel.py - calibrated OCT degradation for the paper's noise study (Step 3).

Calibrated on the real OCTA-500 3 mm B-scans of the 50 test eyes (noise_stats.py, Nov 18):
  log scale                 5.65 grey levels per dB (displayed range about 45 dB)
  noise floor               grey 69 (vitreous median), about 13.5 effective looks, grain 1.6 x 1.3 px (depth x lateral)
  tissue speckle            grain 2.7 x 2.7 px
  retina above floor        about 9.1 dB

Working in linear intensity (floor = 1):  I = 10 ** ((g - FLOOR_GREY) / (10 * GREY_PER_DB))
The local signal above the floor is S = max(median5x5(I) - 1, 0) (a median keeps tissue edges sharp, so the floor next
to tissue is not darkened); w = S / (S + 1) is the tissue share of a pixel.

Two physical degradations, each with its own axis (one-sided, so levels can be compared on one curve):

  1. Fewer averaged frames ("NL<L>": target effective looks L < 13.5).
     Both the tissue speckle and the noise floor become less averaged. Multiplicative mean-1 gamma noise with
     L_add looks is applied, chosen so that the product has L effective looks:
         1/L = 1/L0 + 1/L_add + 1/(L0 L_add)      (L0 = 13.5)
     Tissue pixels get speckle with the tissue grain, floor pixels get noise with the floor grain:
         I' = I * (w * X_tissue + (1 - w) * X_floor)
  2. Signal loss ("NA<D>": D dB less signal, e.g. media opacity, defocus, weaker source; floor unchanged).
         I' = I * (a S + 1) / (S + 1),   a = 10 ** (-D / 10)
     Tissue gets darker towards the floor while its speckle pattern is kept; the floor itself is unchanged.

Correlated gamma fields: Gaussian white noise -> Gaussian blur (sigma per axis) -> gamma quantile transform.
Sigmas come from the measured grain sizes (GRAIN_TISSUE, GRAIN_FLOOR) through a numerically calibrated table
(sigma -> FWHM of log(X), measured exactly as in noise_stats.py).

Back to grey: g' = FLOOR_GREY + 10 * GREY_PER_DB * log10(I'), clipped to 0..255.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.stats import gamma as gamma_dist, norm

GREY_PER_DB = 5.65
FLOOR_GREY = 69.0
L0 = 13.8                        # median variance-based looks of the real floor (noise_stats v4)
# Measured grain sizes (FWHM of the autocorrelation, px, (depth, lateral)) from noise_stats.py v4 (whole-pixel
# flattening). UPDATE these two lines with the values from noise_stats/summary.csv (grain_ax/grain_lat and
# grain_vit_ax/grain_vit_lat, medians) before building test data.
GRAIN_TISSUE = (2.7, 2.7)
GRAIN_FLOOR = (1.6, 1.3)

# FWHM of log(gamma field) vs Gaussian blur sigma, measured with noise_stats.autocorr_fwhm (independent of looks)
_SIG = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30,
        1.40, 1.50]
_FW = [1.01, 1.03, 1.09, 1.20, 1.35, 1.54, 1.78, 2.07, 2.33, 2.55, 2.72, 2.88, 3.03, 3.18, 3.32, 3.64, 3.96, 4.32,
       4.66, 4.98]


def sigma_for_grain(fwhm):
    """Gaussian sigma (px) giving this autocorrelation FWHM; 0 = white noise (FWHM about 1 px)."""
    return 0.0 if fwhm <= 1.02 else float(np.interp(fwhm, _FW, _SIG))


SIGMA_TISSUE = tuple(sigma_for_grain(f) for f in GRAIN_TISSUE)
SIGMA_FLOOR = tuple(sigma_for_grain(f) for f in GRAIN_FLOOR)
SIGNAL_WIN = 5                   # px, median window for the local signal estimate S (edge-preserving)

LOOKS_LEVELS = {"NL8": 8.0, "NL4": 4.0, "NL2": 2.0, "NL1": 1.0}
LOSS_LEVELS = {"NA2": 2.0, "NA4": 4.0, "NA6": 6.0, "NA8": 8.0}
CONDS = {**LOOKS_LEVELS, **LOSS_LEVELS}


def added_looks(L_target, L0=L0):
    """looks of the extra multiplicative noise so that the result has L_target effective looks."""
    if L_target >= L0:
        return np.inf
    return (1.0 + 1.0 / L0) / (1.0 / L_target - 1.0 / L0)


def gamma_field(shape, looks, sigma, rng):
    """mean-1 gamma field with `looks` looks and spatial grain set by sigma (depth, lateral)."""
    g = rng.standard_normal(shape)
    if max(sigma) > 0:
        g = gaussian_filter(g, sigma, mode="reflect")
    g /= g.std() + 1e-12
    u = np.clip(norm.cdf(g), 1e-7, 1 - 1e-7)
    return gamma_dist.ppf(u, looks, scale=1.0 / looks)


def to_linear(grey):
    return 10.0 ** ((grey - FLOOR_GREY) / (10.0 * GREY_PER_DB))


def to_grey(lin):
    return np.clip(FLOOR_GREY + 10.0 * GREY_PER_DB * np.log10(np.maximum(lin, 1e-12)), 0, 255)


def apply(grey, cond, rng):
    """grey: float array 0..255 (any 2D shape). Returns degraded float array 0..255."""
    I = to_linear(np.asarray(grey, dtype=np.float64))
    S = np.maximum(median_filter(I, size=SIGNAL_WIN, mode="reflect") - 1.0, 0.0)
    if cond in LOOKS_LEVELS:
        La = added_looks(LOOKS_LEVELS[cond])
        if not np.isfinite(La):
            return to_grey(I)
        w = S / (S + 1.0)
        Xt = gamma_field(I.shape, La, SIGMA_TISSUE, rng)
        Xf = gamma_field(I.shape, La, SIGMA_FLOOR, rng)
        I2 = I * (w * Xt + (1.0 - w) * Xf)
    elif cond in LOSS_LEVELS:
        a = 10.0 ** (-LOSS_LEVELS[cond] / 10.0)
        I2 = I * (a * S + 1.0) / (S + 1.0)
    else:
        raise ValueError(cond)
    return to_grey(I2)
