#!/usr/bin/env python3
"""
vessel_continuity.py - adjacent B-scan vessel continuity (Nov 17 meeting, item 8).

Idea (instead of averaging neighbouring B-scans): a real vessel crosses many neighbouring B-scans, so in the
en-face vessel map (one row = one B-scan) it continues from row to row. A hallucinated vessel often appears in
one B-scan only. For every vessel pixel we check whether a vessel pixel exists in the B-scan above AND below
(within +-TOL columns, because vessels run at an angle).

Per eye and per run (large vessels and all vessels, en-face slab (304x256)):
  continuity      share of vessel pixels that continue into both neighbouring B-scans   (real OCTA = reference)
  isolated        share of vessel pixels with no vessel in either neighbouring B-scan
  fragments/mm    number of separate vessel pieces per mm of vessel length               (broken vessels)
  gaps            vessel present in B-scans n-1 and n+1 but missing in n, per 100 vessel pixels
  prec_continuous share of continuous generated vessel pixels that lie on a real vessel (1-px tolerance)
  prec_isolated   same for isolated generated vessel pixels
If prec_continuous >> prec_isolated, continuity across B-scans can flag hallucinated vessels.

Run from ~/BBDM (needs eval_volume_lq.py and eval_capdice_lq.py):
    python vessel_continuity.py --pids 10451 --out continuity_test           # 1 eye, ~1 min
    python vessel_continuity.py --workers 8 --out continuity                  # all 50 eyes
Outputs: per_eye.csv, summary.csv, wilcoxon.csv, fig/<eye>_continuity.png
"""
import argparse, csv, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import binary_dilation, label, maximum_filter1d
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_volume_lq import windows, stack, enface_slab, vessel_features
from eval_capdice_lq import NB, C, pred_index, crop_info, ilm_rows

DEFAULT_ARMS = ["ROITOP", "LQAX2", "LQBX2", "LQAX4", "LQBX4"]
LABEL = {"GT": "real OCTA", "ROITOP": "A, HR", "LQAX2": "A, ×2", "LQBX2": "B, ×2", "LQAX4": "A, ×4",
         "LQBX4": "B, ×4", "LQCHR": "C, HR", "LQCX2": "C, ×2", "LQCX4": "C, ×4", "LQN3HR": "2.5D, HR"}
PIX_MM = 9.87e-3   # mm per en-face pixel (3 mm / 304)
TOL = 2            # columns a vessel may shift between neighbouring B-scans


def shift_rows(m, d):
    """m shifted by d rows (rows = B-scans), zero-padded."""
    out = np.zeros_like(m)
    if d > 0:
        out[d:] = m[:-d]
    else:
        out[:d] = m[-d:]
    return out


def continuity(m, gt=None):
    """continuity measures of a boolean vessel map m (304x256); gt for precision split."""
    widen = maximum_filter1d(m.astype(np.uint8), size=2 * TOL + 1, axis=1).astype(bool)
    up, dn = shift_rows(widen, 1), shift_rows(widen, -1)     # vessel in B-scan n-1 / n+1 near this column
    n = max(m.sum(), 1)
    cont = m & up & dn
    iso = m & ~up & ~dn
    gaps = (~m) & shift_rows(m, 1) & shift_rows(m, -1)       # present above and below, missing here
    lab, nfrag = label(m, structure=np.ones((3, 3)))
    length_mm = n * PIX_MM
    out = {"continuity": cont.sum() / n, "isolated": iso.sum() / n,
           "fragments_per_mm": nfrag / max(length_mm, 1e-6), "gaps_per_100px": 100 * gaps.sum() / n}
    if gt is not None:
        near = binary_dilation(gt, np.ones((3, 3)))
        out["prec_continuous"] = (cont & near).sum() / max(cont.sum(), 1)
        out["prec_isolated"] = (iso & near).sum() / max(iso.sum(), 1)
        out["prec_all"] = (m & near).sum() / n
    return out, cont, iso


_G = {}


def one_eye(pid):
    g = _G
    y0 = windows(pid, g["info_top"])
    ilm = np.stack([ilm_rows(pid, k) for k in range(1, NB + 1)])
    ilm = np.clip(np.nan_to_num(ilm, nan=np.nanmedian(ilm)) - y0[:, None], 0, C - 1)
    eg = enface_slab(stack(pid, y0), ilm, g["depth"])
    _, gves, glarge, _ = vessel_features(eg)
    rows, maps = [], {"GT": (eg, gves)}
    for kind, gm in (("large", glarge), ("all", gves)):
        r, _, _ = continuity(gm)
        rows.append({"patient": pid, "method": "GT", "vessels": kind, **r})
    for a in g["arms"]:
        e = enface_slab(stack(pid, y0, a, g["files"][a], g["info"][a]), ilm, g["depth"])
        _, ves, large, _ = vessel_features(e)
        maps[a] = (e, ves)                     # figure shows ALL vessels (large maps are ~0.9 everywhere)
        for kind, m, gm in (("large", large, glarge), ("all", ves, gves)):
            r, _, _ = continuity(m, gm)
            rows.append({"patient": pid, "method": a, "vessels": kind, **r})
    if pid in g["fig_pids"]:
        figure(pid, maps, gves)
    print(f"  done {pid}", flush=True)
    return rows


def figure(pid, maps, gref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["GT"] + [a for a in _G["arms"] if a in maps]
    fig, ax = plt.subplots(2, len(keys), figsize=(2.9 * len(keys), 8.8), dpi=170)
    for j, k in enumerate(keys):
        e, m = maps[k]
        r, cont, iso = continuity(m, None if k == "GT" else gref)
        rgb = np.zeros(m.shape + (3,))
        rgb[m] = (0.55, 0.55, 0.55)            # vessel continuing on one side only
        rgb[cont] = (1.0, 1.0, 1.0)            # continues into both neighbouring B-scans
        rgb[iso] = (0.95, 0.56, 0.17)          # isolated: in this B-scan only
        lo, hi = np.percentile(e, 1), np.percentile(e, 99.5)
        ax[0, j].imshow(np.clip((e - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray")
        ax[0, j].set_title(f"{LABEL.get(k, k)}\n(304×256)", fontsize=9)
        ax[1, j].imshow(rgb, interpolation="nearest")
        ax[1, j].set_title(f"continuity {r['continuity']:.2f}, isolated {r['isolated']:.2f}\n"
                           f"{r['fragments_per_mm']:.1f} fragments/mm", fontsize=8)
        for x in ax[:, j]:
            x.set_xticks([]); x.set_yticks([])
    ax[0, 0].set_ylabel("en-face OCTA", fontsize=9)
    ax[1, 0].set_ylabel("all-vessel map\n(rows = B-scans)", fontsize=9)
    fig.suptitle(f"Eye {pid}: does each vessel continue into the neighbouring B-scans? "
                 "white = continues above and below, grey = one side only, orange = isolated (this B-scan only)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5)
    os.makedirs(_G["fig_dir"], exist_ok=True)
    fig.savefig(os.path.join(_G["fig_dir"], f"{pid}_continuity.png"), facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--pids", nargs="+", type=int, default=None)
    ap.add_argument("--fig_pids", nargs="+", type=int, default=[10451, 10454, 10470, 10492])
    ap.add_argument("--depth", type=int, default=75)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="continuity")
    args = ap.parse_args()
    pids = args.pids or [int(x) for x in open("capdice_patients.txt").read().split()]
    files = {a: pred_index(a) for a in args.arms}
    arms = [a for a in args.arms if files[a] and all(f"{p}_{n:03d}" in files[a] for p in pids
                                                       for n in range(1, NB + 1))]
    if set(arms) != set(args.arms):
        print("skipping incomplete runs:", sorted(set(args.arms) - set(arms)))
    _G.update(arms=arms, files=files, info={a: crop_info(a) for a in arms}, info_top=crop_info("ROITOP"),
              depth=args.depth, fig_pids=set(args.fig_pids), fig_dir=os.path.join(args.out, "fig"))
    os.makedirs(args.out, exist_ok=True)
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            res = pool.map(one_eye, pids)
    else:
        res = [one_eye(p) for p in pids]
    rows = [r for rr in res for r in rr]

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out, "per_eye.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    metrics = [k for k in keys if k not in ("patient", "method", "vessels")]
    summ = []
    for kind in ("large", "all"):
        print(f"\n{kind} vessels, mean over {len(pids)} eyes")
        print(f"{'run':<12}" + "".join(f"{m[:16]:>17}" for m in metrics))
        for a in ["GT"] + arms:
            rs = [r for r in rows if r["method"] == a and r["vessels"] == kind]
            line = {"method": a, "vessels": kind}
            for m in metrics:
                v = np.array([r.get(m, np.nan) for r in rs], float)
                line[m] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
            summ.append(line)
            print(f"{LABEL.get(a, a):<12}" + "".join(f"{line[m]:17.3f}" for m in metrics))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "vessels"] + metrics); w.writeheader(); w.writerows(summ)

    # paired tests: A vs B at x2 and x4; and continuous vs isolated precision within each run
    get = lambda a, kind, m: {r["patient"]: r[m] for r in rows if r["method"] == a and r["vessels"] == kind}
    wr = []
    for x, y in (("LQAX2", "LQBX2"), ("LQAX4", "LQBX4")):
        if x in arms and y in arms:
            for kind in ("large", "all"):
                for m in ("continuity", "isolated", "fragments_per_mm", "gaps_per_100px"):
                    a, b = get(x, kind, m), get(y, kind, m)
                    d = np.array([b[p] - a[p] for p in pids])
                    wr.append({"comparison": f"{y} - {x}", "vessels": kind, "metric": m,
                               "mean_diff": d.mean(), "p": wilcoxon(d).pvalue if np.any(d) else 1.0})
    for a in arms:
        for kind in ("large", "all"):
            c, i = get(a, kind, "prec_continuous"), get(a, kind, "prec_isolated")
            d = np.array([c[p] - i[p] for p in pids])
            wr.append({"comparison": f"{a}: continuous - isolated", "vessels": kind, "metric": "precision",
                       "mean_diff": d.mean(), "p": wilcoxon(d).pvalue if np.any(d) else 1.0})
    with open(os.path.join(args.out, "wilcoxon.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "vessels", "metric", "mean_diff", "p"])
        w.writeheader(); w.writerows(wr)
    print("\npaired tests (Wilcoxon over eyes):")
    for r in wr:
        print(f"  {r['comparison']:<28} {r['vessels']:<6} {r['metric']:<17} diff {r['mean_diff']:+.4f}  p={r['p']:.2g}")
    print(f"\nSaved in {args.out}/  (figures in {args.out}/fig/)")


if __name__ == "__main__":
    main()
