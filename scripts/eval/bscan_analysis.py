#!/usr/bin/env python3
"""
bscan_analysis.py - meeting items of Nov 6 (Dr. Mohammed), using the B-scans that are already generated.
No GPU, no new sampling.

  1. B-scan SSIM / PSNR: every generated (256x256) B-scan patch vs the real OCTA patch at the same place.
  2. Slice consistency: how much neighbouring generated B-scans differ, relative to how much neighbouring
     real B-scans differ (flicker ratio; 1 = like real data, >1 = flickers).
  3. Neighbour-slice averaging (post-processing): average k = 3 or 5 neighbouring generated B-scans
     (slow axis), then recompute B-scan SSIM, en-face large-vessel / capillary Dice, BVC error and the
     stripe ratio. The same averaging is applied to the REAL OCTA as a control (how much averaging alone
     blurs real vessels).
  4. Trace back to B-scans: per B-scan SSIM next to the en-face stripe strength of the same row, saved per
     B-scan; Spearman correlation per eye (do stripes come from bad B-scans?).
  5. Figures for one eye: side-by-side B-scans of every model with local SSIM maps, and en-face local SSIM
     maps.

Sizes: B-scan patch (256x256) = rows y0..y0+255 (ROITOP rule) x columns 24..279.
       Volume per eye (304x256x256) = 304 B-scans. En-face slab image (304x256), 75 px below the ILM.

Run from ~/BBDM (needs eval_volume_lq.py and eval_capdice_lq.py next to it):
    python bscan_analysis.py --pids 10451 --out bscan_test          # one eye, check figures (~2 min)
    python bscan_analysis.py --workers 8 --out bscan_analysis         # all 50 eyes
Outputs in --out:
    per_eye.csv        one row per eye x method x averaging k
    per_bscan.csv      one row per eye x method x B-scan (k = 1): SSIM, PSNR, en-face stripe strength
    summary.csv        mean / std over eyes
    wilcoxon.csv       ROITOP vs each run (k = 1); each run k = 3, 5 vs k = 1
    fig/<eye>_bscans_<n>.png, fig/<eye>_enface_ssim.png
"""
import argparse, csv, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import uniform_filter1d, binary_dilation
from scipy.stats import wilcoxon, spearmanr
from skimage.metrics import structural_similarity
from skimage.morphology import disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_volume_lq import (windows, stack, enface_slab, vessel_features, dice, minmax, VESSEL)
from eval_capdice_lq import (NB, C, ARMS, pred_index, crop_info, ilm_rows, stripe, load01)

DATA = os.environ.get("DATA", "/data/BBDM_data")
DEFAULT_ARMS = ["ROITOP", "LQAX2", "LQAX4", "LQBX2", "LQBX4", "LQCHR", "LQCX2", "LQCX4"]
LABEL = {"ROITOP": "A, HR input", "LQAX2": "A, ×2 input", "LQAX4": "A, ×4 input",
         "LQBX2": "B (ft ×2), ×2 input", "LQBX4": "B (ft ×4), ×4 input", "LQCHR": "C (mixed), HR input",
         "LQCX2": "C (mixed), ×2 input", "LQCX4": "C (mixed), ×4 input", "GT": "real OCTA"}
COND = {"ROITOP": "HR", "LQCHR": "HR", "LQAX2": "X2", "LQBX2": "X2", "LQCX2": "X2",
        "LQAX4": "X4", "LQBX4": "X4", "LQCX4": "X4"}
KS = (1, 3, 5)
RETINA = 160   # retina-only SSIM: rows ILM .. ILM+159 of each A-scan (~500 um: retina, RPE, upper choroid)


# ------------------------------------------------------------------ measures
def retina_mask(ilm, depth=None):
    """(304x256x256) bool: rows ILM .. ILM+depth-1 of every A-scan (ilm: (304x256), window rows)."""
    d = depth or RETINA
    rows = np.arange(C)[None, :, None]
    return (rows >= ilm[:, None, :]) & (rows < ilm[:, None, :] + d)


def bscan_scores(p, g, mask=None):
    """per B-scan SSIM / PSNR over the whole (256x256) patch, and over the retina mask only.
    p, g: (304x256x256) in [0,1]. Returns ss, ps, ss_retina, ps_retina, each (304,)."""
    n = len(g)
    ss, ssr, psr = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        ss[i], m = structural_similarity(g[i], p[i], data_range=1.0, full=True)
        if mask is not None:
            mk = mask[i]
            ssr[i] = m[mk].mean() if mk.any() else np.nan
            mse_r = ((p[i] - g[i])[mk] ** 2).mean() if mk.any() else np.nan
            psr[i] = 10 * np.log10(1.0 / max(mse_r, 1e-12))
    mse = ((p - g) ** 2).reshape(n, -1).mean(1)
    ps = 10 * np.log10(1.0 / np.maximum(mse, 1e-12))
    return ss, ps, ssr, psr


def flicker(v):
    """mean |B-scan i+1 - B-scan i| over the volume."""
    return float(np.abs(np.diff(v, axis=0)).mean())


def avg_slices(v, k):
    return v if k == 1 else uniform_filter1d(v, k, axis=0, mode="nearest")


def row_stripe(e):
    """en-face stripe strength per row (= per B-scan): |row - mean of its two neighbours|, averaged."""
    up = np.vstack([e[1:2], e[:-1]]); dn = np.vstack([e[1:], e[-2:-1]])
    return np.abs(e - (up + dn) / 2).mean(1)


def enface_block(e, gfeat):
    """vessel scores of one en-face image vs the real one."""
    gf, gves, glarge, gcap = gfeat
    f, ves, large, cap = vessel_features(e)
    out = {"large_vessel_dice": dice(large, glarge), "capillary_dice": dice(cap, gcap),
           "vessel_dice": dice(ves, gves)}
    for v in VESSEL:
        out[v + "_abs_err"] = abs(f[v] - gf[v])
    return out, large


# ------------------------------------------------------------------ one eye
def one_eye(pid, arms, depth, files, info, info_top, fig_dir, fig_bscans):
    y0 = windows(pid, info_top)
    ilm = np.stack([ilm_rows(pid, n) for n in range(1, NB + 1)])
    ilm = np.clip(np.nan_to_num(ilm, nan=np.nanmedian(ilm)) - y0[:, None], 0, C - 1)
    g = stack(pid, y0)
    eg = enface_slab(g, ilm, depth)
    gfeat = vessel_features(eg)
    st_g = stripe(eg)
    fl_g = flicker(g)
    rsg = row_stripe(eg)
    mask = retina_mask(ilm)
    rows, brows, vols = [], [], {}

    # control: the same averaging applied to the real OCTA
    for k in KS[1:]:
        ga = avg_slices(g, k)
        e = enface_slab(ga, ilm, depth)
        ss, ps, ssr, psr = bscan_scores(ga, g, mask)
        sc, _ = enface_block(e, gfeat)
        rows.append({"patient": pid, "method": "GT", "k": k, "bscan_ssim": ss.mean(), "bscan_psnr": ps.mean(),
                     "bscan_ssim_retina": np.nanmean(ssr), "bscan_psnr_retina": np.nanmean(psr),
                     "flicker_ratio": flicker(ga) / fl_g, "stripe_ratio": stripe(e) / st_g, **sc})

    for a in arms:
        p = stack(pid, y0, a, files[a], info[a])
        vols[a] = p
        for k in KS:
            pa = avg_slices(p, k)
            e = enface_slab(pa, ilm, depth)
            ss, ps, ssr, psr = bscan_scores(pa, g, mask)
            sc, _ = enface_block(e, gfeat)
            rows.append({"patient": pid, "method": a, "k": k, "bscan_ssim": ss.mean(), "bscan_psnr": ps.mean(),
                         "bscan_ssim_retina": np.nanmean(ssr), "bscan_psnr_retina": np.nanmean(psr),
                         "bscan_ssim_retina_p10": np.nanpercentile(ssr, 10),
                         "flicker_ratio": flicker(pa) / fl_g, "stripe_ratio": stripe(e) / st_g, **sc})
            if k == 1:   # trace back: B-scan quality vs en-face stripe of the same row
                rs = row_stripe(e)
                rows[-1]["rho_ssim_vs_stripe"] = spearmanr(ss, rs).correlation
                rows[-1]["rho_retina_ssim_vs_stripe"] = spearmanr(ssr, rs, nan_policy="omit").correlation
                for i in range(NB):
                    brows.append({"patient": pid, "method": a, "bscan": i + 1, "ssim": ss[i], "psnr": ps[i],
                                  "ssim_retina": ssr[i], "psnr_retina": psr[i],
                                  "enface_row_stripe": rs[i], "enface_row_stripe_gt": rsg[i]})

    if fig_dir is not None:
        figures(pid, g, vols, eg, ilm, depth, fig_dir, fig_bscans, files, mask)
    return rows, brows


# ------------------------------------------------------------------ figures
def input_patch(arm, pid, n):
    """the OCT patch (256x256) the model saw for this B-scan."""
    import glob
    name = f"{pid}_{n:03d}"
    pats = [f"{DATA}/OCTA500_3mm_VOL_LQ{COND.get(arm, 'HR')}_*/test/A/{name}.*"]
    if COND.get(arm, "HR") == "HR":
        pats.append(f"{DATA}/OCTA500_3mm_VOL_ROITOP_*/test/A/{name}.*")
    for pat in pats:
        hit = glob.glob(pat)
        if hit:
            return load01(hit[0])
    return None


def figures(pid, g, vols, eg, ilm, depth, fig_dir, fig_bscans, files, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    arms = [a for a in (_G.get("fig_arms") or vols) if a in vols]
    # B-scans to show: the centre one, plus the B-scan whose ROITOP output has the worst SSIM
    if "ROITOP" in vols:
        ssr_ref = bscan_scores(vols["ROITOP"], g, mask)[2]
        worst = int(np.nanargmin(ssr_ref)) + 1
    else:
        worst = None
    shows = list(fig_bscans) + ([worst] if worst and worst not in fig_bscans else [])
    for n in shows:
        i = n - 1
        nr = len(arms)
        fig, ax = plt.subplots(nr, 4, figsize=(9.6, 2.5 * nr + 0.6), dpi=150)
        ax = np.atleast_2d(ax)
        top = max(float(np.percentile(g[i], 99.5)), 1e-3)   # display: same contrast stretch for real and generated
        for r, a in enumerate(arms):
            p = vols[a][i]
            s, m = structural_similarity(g[i], p, data_range=1.0, full=True)
            sr = m[mask[i]].mean()
            inp = input_patch(a, pid, n)
            tiles = [(inp if inp is not None else np.zeros_like(p), "OCT input", "gray", 0, 1),
                     (p, f"generated OCTA", "gray", 0, top),
                     (g[i], "real OCTA", "gray", 0, top),
                     (m, f"local SSIM", "magma", 0, 1)]
            for c, (im, t, cm, lo, hi) in enumerate(tiles):
                ax[r, c].imshow(im, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
                if r == 0:
                    ax[r, c].set_title(f"{t}\n(256×256)" if c < 3 else "local SSIM map\n(256×256)", fontsize=9)
                if c == 3:   # retina region outlined; numbers: whole patch / retina only
                    ax[r, c].contour(mask[i], levels=[0.5], colors="#3ddc97", linewidths=0.8)
                    ax[r, c].text(0.98, 0.03, f"patch {s:.3f}\nretina {sr:.3f}", transform=ax[r, c].transAxes,
                                  ha="right", va="bottom", fontsize=7.5, color="white",
                                  bbox=dict(facecolor="black", alpha=0.55, pad=1.5, edgecolor="none"))
            ax[r, 0].set_ylabel(LABEL.get(a, a), fontsize=9)
        tag = " (worst ROITOP B-scan)" if n == worst else ""
        fig.suptitle(f"Eye {pid}, B-scan {n}{tag}: OCT input, generated and real OCTA, local SSIM "
                     f"(7×7 window; bright = similar).\nGreen outline = retina region used for retina SSIM "
                     f"(ILM to ILM+{RETINA} px); OCTA shown contrast-stretched (same scale for real and generated).",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(fig_dir, f"{pid}_bscans_{n:03d}.png"), facecolor="white")
        plt.close(fig)

    # en-face local SSIM maps
    nc = len(arms) + 1
    fig, ax = plt.subplots(2, nc, figsize=(2.1 * nc, 5.6), dpi=150)
    ax[0, 0].imshow(eg, cmap="gray", vmin=0, vmax=1); ax[0, 0].set_title("real OCTA\n(304×256)", fontsize=9)
    ax[1, 0].axis("off")
    for c, a in enumerate(arms, start=1):
        e = enface_slab(vols[a], ilm, depth)
        s, m = structural_similarity(eg, e, data_range=1.0, full=True)
        ax[0, c].imshow(e, cmap="gray", vmin=0, vmax=1)
        ax[0, c].set_title(f"{LABEL.get(a, a)}\n(304×256)", fontsize=8)
        ax[1, c].imshow(m, cmap="magma", vmin=0, vmax=1)
        ax[1, c].set_title(f"local SSIM, mean {s:.3f}", fontsize=8)
    for x in ax.ravel():
        x.set_xticks([]); x.set_yticks([])
    ax[0, 0].set_ylabel(f"en-face, {depth} px slab", fontsize=9)
    ax[1, 1].set_ylabel("local SSIM map", fontsize=9)
    fig.suptitle(f"Eye {pid}: en-face images and where they differ from the real OCTA "
                 "(bright = similar, dark = different)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(fig_dir, f"{pid}_enface_ssim.png"), facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ main
_G = {}


def _work(pid):
    a = _G
    fig_dir = a["fig_dir"] if pid in a["fig_pids"] else None
    out = one_eye(pid, a["arms"], a["depth"], a["files"], a["info"], a["info_top"], fig_dir, a["fig_bscans"])
    print(f"  done {pid}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--pids", nargs="+", type=int, default=None, help="default: capdice_patients.txt")
    ap.add_argument("--depth", type=int, default=75, help="slab depth below the ILM (matched value: 75)")
    ap.add_argument("--fig_pids", nargs="+", type=int, default=None, help="eyes to draw (default: first eye)")
    ap.add_argument("--fig_bscans", nargs="+", type=int, default=[152], help="B-scan numbers 1..304 to draw")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fig_arms", nargs="+", default=None, help="runs to draw (default: all)")
    ap.add_argument("--retina", type=int, default=160, help="retina SSIM depth below the ILM (px)")
    ap.add_argument("--out", default="bscan_analysis")
    args = ap.parse_args()

    pids = args.pids or [int(x) for x in open("capdice_patients.txt").read().split()]
    files = {a: pred_index(a) for a in args.arms}
    have = lambda a: all(f"{p}_{n:03d}" in files[a] for p in pids for n in range(1, NB + 1))
    arms = [a for a in args.arms if files[a] and have(a)]
    skipped = [a for a in args.arms if a not in arms]
    if skipped:
        print(f"skipping incomplete / missing runs: {skipped}")
    if not arms:
        sys.exit("no complete runs found")
    print(f"runs: {arms}\neyes: {len(pids)}   averaging k: {KS}   slab depth: {args.depth} px")
    info = {a: crop_info(a) for a in arms}
    info_top = info.get("ROITOP") or crop_info("ROITOP")
    os.makedirs(args.out, exist_ok=True)
    _G.update(arms=arms, depth=args.depth, files=files, info=info, info_top=info_top,
              fig_dir=os.path.join(args.out, "fig"), fig_pids=set(args.fig_pids or pids[:1]),
              fig_bscans=args.fig_bscans, fig_arms=args.fig_arms)
    global RETINA
    RETINA = args.retina

    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            res = pool.map(_work, pids)
    else:
        res = [_work(p) for p in pids]
    rows = [r for rr, _ in res for r in rr]
    brows = [b for _, bb in res for b in bb]

    def write(path, rs):
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rs)
    write(os.path.join(args.out, "per_eye.csv"), rows)
    write(os.path.join(args.out, "per_bscan.csv"), brows)

    metrics = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in ("patient", "method", "k")]
    summ = []
    print("\nmean over eyes (k = number of neighbouring B-scans averaged; GT rows = averaging the real OCTA)")
    print(f"{'method':<8}{'k':>2}  " + "  ".join(f"{m[:14]:>14}" for m in metrics))
    for a in ["GT"] + arms:
        for k in KS:
            rs = [r for r in rows if r["method"] == a and r["k"] == k]
            if not rs:
                continue
            line = {"method": a, "k": k, "eyes": len(rs)}
            for m in metrics:
                v = np.array([r.get(m, np.nan) for r in rs], float)
                if np.all(np.isnan(v)):
                    continue
                line[m + "_mean"] = float(np.nanmean(v))
                line[m + "_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
            summ.append(line)
            print(f"{a:<8}{k:>2}  " + "  ".join(f"{line.get(m + '_mean', np.nan):14.4f}" for m in metrics))
    write(os.path.join(args.out, "summary.csv"), summ)

    # paired tests over eyes
    LOWER = {v + "_abs_err" for v in VESSEL} | {"flicker_ratio"}
    tests = ["bscan_ssim", "bscan_ssim_retina", "bscan_psnr", "bscan_psnr_retina", "large_vessel_dice", "capillary_dice", "BVC_abs_err",
             "flicker_ratio", "stripe_ratio"]
    get = lambda a, k, m: {r["patient"]: r.get(m, np.nan) for r in rows if r["method"] == a and r["k"] == k}
    wr = []
    if len(pids) >= 6:
        pairs = [(("ROITOP", 1), (a, 1)) for a in arms if a != "ROITOP"] + \
                [((a, 1), (a, k)) for a in arms for k in KS[1:]]
        for (a1, k1), (a2, k2) in pairs:
            block = []
            for m in tests:
                x, y = get(a1, k1, m), get(a2, k2, m)
                ps_ = [p for p in pids if p in x and p in y]
                xv, yv = np.array([x[p] for p in ps_], float), np.array([y[p] for p in ps_], float)
                if m in ("flicker_ratio", "stripe_ratio"):          # distance from real data (ratio 1)
                    xv, yv = np.abs(np.log(xv)), np.abs(np.log(yv))
                try:
                    pv = float(wilcoxon(xv, yv).pvalue)
                except ValueError:
                    pv = float("nan")
                second_better = yv.mean() < xv.mean() if (m in LOWER or "ratio" in m) else yv.mean() > xv.mean()
                block.append({"first": f"{a1} k={k1}", "second": f"{a2} k={k2}", "metric": m,
                              "first_mean": float(np.mean(xv)), "second_mean": float(np.mean(yv)),
                              "p": pv, "second_better": bool(second_better)})
            order = sorted([(i, b["p"]) for i, b in enumerate(block) if b["p"] == b["p"]], key=lambda t: t[1])
            run = 0.0
            for j, (i, pv) in enumerate(order):
                run = max(run, min(1.0, (len(order) - j) * pv)); block[i]["p_holm"] = run
            wr += block
        write(os.path.join(args.out, "wilcoxon.csv"), wr)
        print("\nWilcoxon (paired over eyes, Holm within each comparison): see wilcoxon.csv")
        for b in wr:
            if b["metric"] in ("bscan_ssim_retina", "large_vessel_dice", "stripe_ratio"):
                lab = "|log stripe_ratio|" if b["metric"] == "stripe_ratio" else b["metric"]
                print(f"  {b['first']:<12} vs {b['second']:<12} {lab:<19} "
                      f"{b['first_mean']:.4f} -> {b['second_mean']:.4f}  p_holm={b.get('p_holm', np.nan):.2g}")
    print(f"\nSaved in {args.out}/  (figures in {args.out}/fig/)")


if __name__ == "__main__":
    main()
