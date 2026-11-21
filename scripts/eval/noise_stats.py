#!/usr/bin/env python3
"""
noise_stats.py - Step 1 of the noise study: measure the real noise statistics of the OCTA-500 3 mm OCT B-scans.

Why: the paper's noise model must be calibrated to real data. This script measures, on the raw B-scans of the
50 test eyes (columns 24-279 = the ROITOP crop columns), per B-scan and per eye:

  regions (rows are found from the ILM, per column):
    vitreous  = ILM-60 .. ILM-15 px   (no tissue: pure noise floor)
    retina    = ILM+10 .. ILM+150 px  (inner + middle retina, flattened along the ILM)
    deep      = bottom rows, at least 350 px below the ILM (far below the choroid; often noise only)

  measured
    *_med, *_std        median and standard deviation of grey levels (0-255) in each region
    vit_clip_frac       share of vitreous pixels at the lowest grey level (noise floor clipped by the export?)
    snr_grey            retina median - vitreous median, in grey levels
    looks_vit_std       effective looks of the floor from its std and the common scale (variance-based; use this
                        one to compare with simulated levels, it stays valid when several noise sources add)
    grain_vit_ax/_lat   the same for the vitreous noise floor (pure noise, no tissue layers)
    grain_ax, grain_lat speckle grain size = full width at half maximum (px) of the autocorrelation of the
                        high-pass retina band, along depth (axial) and across A-scans (lateral)
    looks_*             effective number of looks L (1 = raw single-frame speckle, larger = averaged / smoother),
                        from the SKEWNESS of the log-grey values; skewness of log-gamma noise depends only on L,
                        not on the unknown log-compression scale
    grey_per_db_*       grey levels per dB of the log compression, from std of log values and L:
                        std_dB = (10 / ln 10) * sqrt(trigamma(L))
    range_db            implied displayed dynamic range = 255 / grey_per_db
    snr_db              snr_grey / grey_per_db = retina signal above the noise floor, in dB
  Which estimate is used: the vitreous estimate when the noise floor is not clipped (< 0.1% of pixels at the
  lowest grey level), else the retina estimate. In OCTA-500 the floor sits near grey 69 and is never clipped, while
  the retina texture has POSITIVE skew (tissue structure, not pure speckle), so the retina estimate is undefined.
  The log compression is one fixed export mapping for all scans, so one common scale is used for every eye:
    grey_per_db_common = median over eyes of grey_per_db_used
    snr_db = snr_grey / grey_per_db_common,  range_db = 255 / grey_per_db_common

Run from ~/BBDM (uses ilm_rows from eval_capdice_lq.py):
    python noise_stats.py --pids 10451 --out noise_stats_test          # 1 eye, ~1 min: check it works
    python noise_stats.py --workers 8 --out noise_stats                 # 50 test eyes, ~10-20 min, CPU only
Outputs (in --out):
    per_bscan.csv         one row per analysed B-scan
    per_eye.csv           one row per eye
    summary.csv           mean / std / min / max over eyes of every per-eye value
    fig/<eye>_noise.png   B-scan with regions, autocorrelation and grey-level histograms
"""
import argparse, csv, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter
from scipy.optimize import brentq
from scipy.special import polygamma
from scipy.stats import skew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_capdice_lq import ilm_rows, RAW, X0, C   # noqa: E402

NB = 304                     # B-scans per eye
VIT = (-60, -15)             # vitreous rows relative to the ILM
RET = (10, 150)              # retina rows relative to the ILM
DEEP_MIN = 350               # deep region starts at least this far below the ILM
DEEP_ROWS = 40               # and uses at most the bottom 40 rows
HP_SIGMA = 4.0               # high-pass: subtract a Gaussian blur of this sigma (px) before texture statistics
DB = 10.0 / np.log(10.0)


# ------------------------------------------------------------------ statistics helpers
def looks_from_skew(s):
    """effective looks L of gamma speckle whose log has skewness s (s < 0); nan if not invertible."""
    f = lambda L: polygamma(2, L) / polygamma(1, L) ** 1.5 - s
    try:
        return float(brentq(f, 0.05, 1e4))
    except (ValueError, RuntimeError):
        return float("nan")


def looks_from_std(std_grey, gpd):
    """effective looks from the std of log-grey values when the log scale gpd (grey per dB) is known."""
    v = (std_grey / (gpd * DB)) ** 2
    try:
        return float(brentq(lambda L: polygamma(1, L) - v, 0.05, 1e5))
    except (ValueError, RuntimeError):
        return float("nan")


def grey_per_db(std_grey, L):
    return std_grey / (DB * np.sqrt(polygamma(1, L))) if np.isfinite(L) else float("nan")


def fwhm(profile):
    """full width at half maximum of a symmetric profile centred at index len//2 (linear interpolation)."""
    c = len(profile) // 2
    p = profile / profile[c]
    right = p[c:]
    k = np.argmax(right < 0.5)
    if k == 0:
        return float("nan")
    x = (k - 1) + (right[k - 1] - 0.5) / (right[k - 1] - right[k])
    return 2.0 * x


def autocorr_fwhm(band):
    """axial and lateral FWHM (px) of the normalised 2D autocorrelation of a zero-mean band."""
    b = band - band.mean()
    F = np.fft.fft2(b, s=(2 * b.shape[0], 2 * b.shape[1]))
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2)))
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    return fwhm(ac[cy - 20:cy + 21, cx]), fwhm(ac[cy, cx - 20:cx + 21]), ac[cy - 12:cy + 13, cx - 12:cx + 13]


def flatten(img, ilm, lo, hi):
    """rows ilm+lo .. ilm+hi-1 of every column, as a (hi-lo, ncols) band following the ILM.
    Whole-pixel shifts only (ILM rounded): interpolating at fractional rows would blur the noise axially and
    inflate the measured grain size."""
    rows = np.round(ilm).astype(int)[None, :] + np.arange(lo, hi)[:, None]
    cols = np.broadcast_to(np.arange(img.shape[1])[None, :], rows.shape)
    ok = (rows >= 0) & (rows <= img.shape[0] - 1)
    band = img[np.clip(rows, 0, img.shape[0] - 1), cols]
    return band, ok


# ------------------------------------------------------------------ one B-scan
def bscan_stats(pid, n, transform=None):
    """transform(img, pid, n) -> img: optional degradation applied to the crop before measuring (noise_check.py)."""
    a = np.asarray(Image.open(f"{RAW}/OCT/{pid}/{n}.bmp").convert("L"), dtype=np.float64)
    img = a[:, X0:X0 + C]
    if transform is not None:
        img = np.asarray(transform(img, pid, n), dtype=np.float64)
    ilm = ilm_rows(pid, n)
    if np.all(np.isnan(ilm)):
        return None
    ilm = np.nan_to_num(ilm, nan=np.nanmedian(ilm))
    ilm = median_filter(ilm, size=41, mode="nearest")    # remove ILM-detection jumps (cysts, foveal pit errors)
    H = img.shape[0]
    vit, vok = flatten(img, ilm, *VIT)
    ret, rok = flatten(img, ilm, *RET)
    vit, ret = vit[vok], ret[rok]
    deep_lo = int(max(np.max(ilm) + DEEP_MIN, H - DEEP_ROWS))
    deep = img[deep_lo:] if deep_lo < H else np.empty((0, C))

    band, bok = flatten(img, ilm, *RET)
    hp = band - gaussian_filter(band, HP_SIGMA)
    hp = hp[:, bok.all(axis=0)] if bok.all(axis=0).sum() > 32 else hp
    g_ax, g_lat, ac = autocorr_fwhm(hp)
    vband = flatten(img, ilm, *VIT)[0]
    gv_ax, gv_lat, _ = autocorr_fwhm(vband - vband.mean(axis=1, keepdims=True))   # grain of the pure noise floor

    vmin = img.min()
    out = {"patient": pid, "bscan": n,
           "vit_med": np.median(vit) if vit.size else np.nan, "vit_std": vit.std() if vit.size else np.nan,
           "vit_clip_frac": float(np.mean(vit <= vmin)) if vit.size else np.nan,
           "vit_skew": float(skew(vit.ravel())) if vit.size > 50 else np.nan,
           "ret_med": np.median(ret), "ret_std": ret.std(),
           "ret_hp_std": hp.std(), "ret_hp_skew": float(skew(hp.ravel())),
           "deep_med": np.median(deep) if deep.size else np.nan, "deep_std": deep.std() if deep.size else np.nan,
           "grain_ax": g_ax, "grain_lat": g_lat, "grain_vit_ax": gv_ax, "grain_vit_lat": gv_lat}
    out["snr_grey"] = out["ret_med"] - out["vit_med"]
    return out, (img, ilm, deep_lo, ac, vit, hp)


# ------------------------------------------------------------------ one eye
_G = {}


def one_eye(pid):
    rows, keep = [], None
    for n in range(1, NB + 1, _G["step"]):
        r = bscan_stats(pid, n)
        if r is None:
            continue
        rows.append(r[0])
        if keep is None and n >= NB // 2:
            keep = r[1]                                   # a central B-scan for the figure
    if not rows:
        print(f"  {pid}: no usable B-scans", flush=True)
        return [], None
    med = lambda k: float(np.nanmedian([r[k] for r in rows]))
    e = {"patient": pid, "bscans": len(rows)}
    for k in ("vit_med", "vit_std", "vit_clip_frac", "vit_skew", "ret_med", "ret_std", "ret_hp_std", "ret_hp_skew",
              "deep_med", "deep_std", "grain_ax", "grain_lat", "grain_vit_ax", "grain_vit_lat", "snr_grey"):
        e[k] = med(k)
    e["looks_vit"] = looks_from_skew(e["vit_skew"]) if e["vit_clip_frac"] < 0.001 else float("nan")
    e["looks_ret"] = looks_from_skew(e["ret_hp_skew"])
    e["grey_per_db_vit"] = grey_per_db(e["vit_std"], e["looks_vit"])
    e["grey_per_db_ret"] = grey_per_db(e["ret_hp_std"], e["looks_ret"])
    gpd = e["grey_per_db_vit"] if np.isfinite(e["grey_per_db_vit"]) else e["grey_per_db_ret"]
    e["grey_per_db_used"] = gpd
    e["snr_db_own"] = e["snr_grey"] / gpd if np.isfinite(gpd) and gpd > 0 else float("nan")
    if pid in _G["fig_pids"] and keep is not None:
        figure(pid, e, keep)
    print(f"  done {pid}: SNR {e['snr_grey']:.0f} grey levels ({e['snr_db_own']:.1f} dB with own scale), grain retina {e['grain_ax']:.1f} x {e['grain_lat']:.1f} px, "
          f"vitreous {e['grain_vit_ax']:.1f} x {e['grain_vit_lat']:.1f} px, "
          f"looks(ret) {e['looks_ret']:.2f}, vitreous clipped {100 * e['vit_clip_frac']:.0f}%", flush=True)
    return rows, e


def figure(pid, e, keep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img, ilm, deep_lo, ac, vit, hp = keep
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.1, 0.8, 1, 1]})
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
    x = np.arange(img.shape[1])
    for lo, hi, col, lab in ((VIT[0], VIT[1], "tab:blue", "vitreous"), (RET[0], RET[1], "tab:green", "retina")):
        ax[0].plot(x, ilm + lo, color=col, lw=0.8, label=lab); ax[0].plot(x, ilm + hi, color=col, lw=0.8)
    if deep_lo < img.shape[0]:
        ax[0].axhline(deep_lo, color="tab:orange", lw=0.8, label="deep")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_title(f"Eye {pid}, central B-scan, crop columns\n(640×256)", fontsize=9)
    ax[1].imshow(ac / ac.max(), cmap="magma", extent=(-12.5, 12.5, 12.5, -12.5))
    ax[1].set_title(f"autocorrelation of retina texture\n(25×25 px), FWHM {e['grain_ax']:.1f} (depth) "
                    f"× {e['grain_lat']:.1f} px", fontsize=9)
    ax[1].set_xlabel("lateral shift (px)"); ax[1].set_ylabel("axial shift (px)")
    ax[2].hist(vit.ravel(), bins=np.arange(0, 257, 2), color="tab:blue", alpha=0.7, density=True, label="vitreous")
    ax[2].hist(img.ravel(), bins=np.arange(0, 257, 2), color="0.6", alpha=0.5, density=True, label="whole B-scan")
    ax[2].set_title(f"grey levels; vitreous clipped {100 * e['vit_clip_frac']:.0f}%", fontsize=9)
    ax[2].legend(fontsize=7); ax[2].set_xlabel("grey level")
    ax[3].hist(hp.ravel(), bins=80, color="tab:green", alpha=0.8, density=True)
    ax[3].set_title(f"high-pass retina texture: skew {e['ret_hp_skew']:.2f}\n"
                    f"vitreous: L ≈ {e['looks_vit']:.1f}, {e['grey_per_db_used']:.2f} grey/dB; SNR {e['snr_grey']:.0f} grey levels", fontsize=9)
    ax[3].set_xlabel("grey level − local mean")
    fig.tight_layout()
    os.makedirs(_G["fig_dir"], exist_ok=True)
    fig.savefig(os.path.join(_G["fig_dir"], f"{pid}_noise.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main
def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", nargs="+", type=int, default=None)
    ap.add_argument("--pids_file", default="capdice_patients.txt")
    ap.add_argument("--step", type=int, default=4, help="analyse every n-th B-scan (4 -> 76 per eye)")
    ap.add_argument("--fig_pids", nargs="+", type=int, default=[10451, 10454, 10470, 10492])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="noise_stats")
    args = ap.parse_args()
    pids = args.pids or [int(x) for x in open(args.pids_file).read().split()]
    _G.update(step=args.step, fig_pids=set(args.fig_pids), fig_dir=os.path.join(args.out, "fig"))
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(pids)} eyes, every {args.step}th B-scan, columns {X0}-{X0 + C - 1}")
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            res = pool.map(one_eye, pids)
    else:
        res = [one_eye(p) for p in pids]
    bs = [r for rr, _ in res for r in rr]
    eyes = [e for _, e in res if e is not None]
    g = np.array([e["grey_per_db_used"] for e in eyes], float)
    gpd_c = float(np.nanmedian(g)) if np.isfinite(g).any() else float("nan")
    for e in eyes:
        e["grey_per_db_common"] = gpd_c
        e["looks_vit_std"] = looks_from_std(e["vit_std"], gpd_c)
        e["range_db"] = 255.0 / gpd_c
        e["snr_db"] = e["snr_grey"] / gpd_c
    for r in bs:
        r["snr_db"] = r["snr_grey"] / gpd_c
    print(f"\ncommon log scale: {gpd_c:.2f} grey levels per dB (median of {int(np.isfinite(g).sum())} eyes), "
          f"displayed range {255.0 / gpd_c:.1f} dB")
    write_csv(os.path.join(args.out, "per_bscan.csv"), bs)
    write_csv(os.path.join(args.out, "per_eye.csv"), eyes)
    keys = [k for k in eyes[0] if k not in ("patient", "bscans")]
    summ = []
    for k in keys:
        v = np.array([e[k] for e in eyes], float)
        summ.append({"measure": k, "mean": np.nanmean(v), "median": np.nanmedian(v), "std": np.nanstd(v),
                     "min": np.nanmin(v),
                     "max": np.nanmax(v), "eyes": int(np.isfinite(v).sum())})
    write_csv(os.path.join(args.out, "summary.csv"), summ)
    print(f"\nsummary over {len(eyes)} eyes")
    for s in summ:
        print(f"  {s['measure']:<18} mean {s['mean']:9.3f}  median {s['median']:9.3f}  std {s['std']:8.3f}  "
              f"range {s['min']:9.3f} .. {s['max']:9.3f}  ({s['eyes']} eyes)")
    print(f"\nSaved in {args.out}/  (figures in {args.out}/fig/)")


if __name__ == "__main__":
    main()
