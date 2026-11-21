#!/usr/bin/env python3
"""
make_tables.py - rebuild every summary number in the report and presentation from the per-eye evaluation CSVs.

Inputs (from eval_volume_lq.py, slab en-face rows are used):
  --main   per-eye CSV of the main runs        (default results/per_eye_all_runs.csv)
  --cross  per-eye CSV of the cross-resolution runs (default results/per_eye_cross_resolution.csv)
Outputs in --out (default results/):
  summary_medians.csv            median per run: Dice, width error, SSIM, PSNR, LPIPS (+ mean large-vessel Dice)
  delta_dice_B_minus_A_per_eye.csv   per eye: ResAdapt − FullRes at x2 and x4, large vessels and capillaries
  comparisons.csv                paired comparisons: median Δ, eyes improved, Wilcoxon p, gap closed
  outlier_check.csv              medians with and without eye 10454 (vitreous opacities)
  trend_check.csv                does fine-tuning help more on eyes FullRes already does well? (Spearman)
  ../figures/delta_dice_heatmap.png  per-eye heatmap of ResAdapt − FullRes

Run from the repository root:  python scripts/analysis/make_tables.py

Model names: FullRes-BBDM (run ROITOP / LQAX2 / LQAX4), ResAdapt-BBDM (LQBX2, LQBX4),
MultiRes-BBDM (LQCHR, LQCX2, LQCX4), cross-resolution ResAdapt-x2 on x4 (LQBX2X4), ResAdapt-x4 on x2 (LQBX4X2).
"""
import argparse, csv, os
import numpy as np
from scipy.stats import wilcoxon, spearmanr

NAMES = {"ROITOP": "FullRes-BBDM, full-resolution input (upper reference)", "LQAX2": "FullRes-BBDM, x2 input",
         "LQAX4": "FullRes-BBDM, x4 input", "LQBX2": "ResAdapt-x2, x2 input", "LQBX4": "ResAdapt-x4, x4 input",
         "LQCHR": "MultiRes-BBDM, full-resolution input", "LQCX2": "MultiRes-BBDM, x2 input",
         "LQCX4": "MultiRes-BBDM, x4 input", "LQBX2X4": "ResAdapt-x2, x4 input (cross-resolution)",
         "LQBX4X2": "ResAdapt-x4, x2 input (cross-resolution)", "NOISE": "chance: random-noise image",
         "OTHER_EYE": "chance: another eye's OCTA"}
METRICS = ["large_vessel_dice", "capillary_dice", "BVC_abs_err", "ssim3d", "enface_ssim", "enface_psnr", "enface_lpips"]
OUTLIER = "10454"


def load(paths):
    rows = []
    for p in paths:
        if p and os.path.exists(p):
            rows += [r for r in csv.DictReader(open(p)) if r.get("projection", "slab") == "slab"]
    return rows


def per_eye(rows, run, metric):
    return {r["patient"]: float(r[metric]) for r in rows if r["method"] == run and r.get(metric, "") != ""}


def write(path, out):
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/per_eye_all_runs.csv")
    ap.add_argument("--cross", default="results/per_eye_cross_resolution.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figdir", default="figures")
    a = ap.parse_args()
    rows = load([a.main])
    rows += [r for r in load([a.cross]) if r["method"] in ("LQBX2X4", "LQBX4X2")]
    runs = [m for m in NAMES if any(r["method"] == m for r in rows)]

    # 1. medians per run
    summ = []
    for m in runs:
        d = {"run": m, "model": NAMES[m]}
        for k in METRICS:
            v = list(per_eye(rows, m, k).values())
            d[k + "_median"] = round(float(np.median(v)), 4) if v else ""
        lv = list(per_eye(rows, m, "large_vessel_dice").values())
        d["large_vessel_dice_mean"] = round(float(np.mean(lv)), 4)
        d["n_eyes"] = len(lv)
        summ.append(d)
    write(os.path.join(a.out, "summary_medians.csv"), summ)

    # 2. per-eye ResAdapt − FullRes
    keys = [("large", "large_vessel_dice"), ("cap", "capillary_dice")]
    eyes = sorted(per_eye(rows, "LQAX2", "large_vessel_dice"))
    de = []
    for e in eyes:
        r = {"eye": e}
        for short, k in keys:
            for f in ("x2", "x4"):
                A = per_eye(rows, f"LQA{f.upper()}", k); B = per_eye(rows, f"LQB{f.upper()}", k)
                if e in A and e in B:
                    r[f"{short}_A_{f}"] = round(A[e], 4); r[f"{short}_B_{f}"] = round(B[e], 4)
                    r[f"{short}_delta_{f}"] = round(B[e] - A[e], 4)
        de.append(r)
    write(os.path.join(a.out, "delta_dice_B_minus_A_per_eye.csv"), de)

    # 3. paired comparisons + gap closed
    pairs = [("LQBX2", "LQAX2", "ResAdapt-x2 vs FullRes, x2"), ("LQBX4", "LQAX4", "ResAdapt-x4 vs FullRes, x4"),
             ("LQCX2", "LQAX2", "MultiRes vs FullRes, x2"), ("LQCX4", "LQAX4", "MultiRes vs FullRes, x4"),
             ("LQBX2X4", "LQAX4", "ResAdapt-x2 on x4 vs FullRes, x4"), ("LQBX2X4", "LQBX4", "ResAdapt-x2 on x4 vs ResAdapt-x4"),
             ("LQBX4X2", "LQAX2", "ResAdapt-x4 on x2 vs FullRes, x2")]
    ref = {"x2": "LQAX2", "x4": "LQAX4"}
    comp = []
    for b, aa, lab in pairs:
        if b not in runs or aa not in runs:
            continue
        for _, k in keys:
            A, B = per_eye(rows, aa, k), per_eye(rows, b, k)
            common = sorted(set(A) & set(B))
            d = np.array([B[e] - A[e] for e in common])
            r = {"comparison": lab, "metric": k, "n": len(d), "median_delta": round(float(np.median(d)), 4),
                 "mean_delta": round(float(np.mean(d)), 4), "eyes_improved": int((d > 0).sum()),
                 "p_wilcoxon": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0}
            f = "x4" if aa in ("LQAX4", "LQBX4") else "x2"
            if aa == ref[f] and "ROITOP" in runs:
                H = per_eye(rows, "ROITOP", k)
                gap = np.median([H[e] - A[e] for e in common if e in H])
                r["median_gap_to_full_res"] = round(float(gap), 4)
                r["gap_closed_percent"] = round(100 * float(np.median(d)) / gap, 1) if gap > 0 else ""
            comp.append(r)
    write(os.path.join(a.out, "comparisons.csv"), comp)

    # 4. outlier check
    oc = []
    for m in runs:
        v = per_eye(rows, m, "large_vessel_dice")
        oc.append({"run": m, "median_all": round(float(np.median(list(v.values()))), 4),
                   f"median_without_{OUTLIER}": round(float(np.median([x for e, x in v.items() if e != OUTLIER])), 4),
                   f"eye_{OUTLIER}": round(v.get(OUTLIER, float("nan")), 4)})
    write(os.path.join(a.out, "outlier_check.csv"), oc)

    # 5. trend: FullRes Dice vs (FullRes − ResAdapt)
    tc = []
    for f in ("x2", "x4"):
        A = per_eye(rows, f"LQA{f.upper()}", "large_vessel_dice"); B = per_eye(rows, f"LQB{f.upper()}", "large_vessel_dice")
        for drop in (False, True):
            es = [e for e in A if e in B and not (drop and e == OUTLIER)]
            x = np.array([A[e] for e in es]); y = np.array([A[e] - B[e] for e in es])
            rho, p = spearmanr(x, y)
            tc.append({"input": f, "excluding_10454": drop, "n": len(es), "spearman_rho": round(float(rho), 3), "p": round(float(p), 4)})
    write(os.path.join(a.out, "trend_check.csv"), tc)

    # 6. heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rowsH = [("large", "x2"), ("large", "x4"), ("cap", "x2"), ("cap", "x4")]
    order = sorted(de, key=lambda r: r.get("large_delta_x4", 0))
    M = np.array([[r.get(f"{s}_delta_{f}", np.nan) for r in order] for s, f in rowsH])
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(16, 3.6), dpi=200)
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(4)); ax.set_yticklabels(["large vessels, x2", "large vessels, x4", "capillaries, x2", "capillaries, x4"])
    ax.set_xticks(range(len(order))); ax.set_xticklabels([r["eye"] for r in order], rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, label="ΔDice (ResAdapt − FullRes)")
    ax.set_title("Per-eye ΔDice, ResAdapt-BBDM − FullRes-BBDM, same input; blue = ResAdapt better, red = worse; "
                 "eyes sorted by large-vessel Δ at x4; en-face slab (304×256)", fontsize=9, loc="left")
    fig.tight_layout()
    os.makedirs(a.figdir, exist_ok=True)
    fig.savefig(os.path.join(a.figdir, "delta_dice_heatmap.png"), facecolor="white")
    print("wrote", os.path.join(a.figdir, "delta_dice_heatmap.png"))

    for r in comp:
        if r["metric"] == "large_vessel_dice":
            print(f"  {r['comparison']:<36} median Δ {r['median_delta']:+.4f}  improved {r['eyes_improved']}/{r['n']}  "
                  f"p={r['p_wilcoxon']:.2g}  gap closed {r.get('gap_closed_percent', '')}")


if __name__ == "__main__":
    main()
