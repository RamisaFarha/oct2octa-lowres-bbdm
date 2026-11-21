#!/usr/bin/env python3
"""
bg_check.py - is the "missed top edge" just the faint background haze of the generated OCTA?

For every test eye and run, the generated B-scans (256x256) are compared with the real OCTA after setting
generated pixels below T grey levels (out of 255) to 0, for T = 0 (as generated), 3 and 5.
If the low-SSIM band at the top and part of B's SSIM gain come from the haze, they shrink once it is removed.

Measured per eye (median over 304 B-scans):
  ssim_patch        SSIM over the whole (256x256) patch
  ssim_retina       SSIM inside the retina (ILM .. ILM+160 px)
  ssim_vitreous     SSIM above the retina (rows < ILM-3)
  ssim_top20        SSIM in the top 20 rows of the patch
  bg_grey           mean generated grey (0-255) above the retina where the real OCTA is 0 (T = 0 only)
  real_dim_frac     share of real retina pixels below T (how much real signal T would remove; T > 0)
  gen_dim_frac      share of generated retina pixels below T
Plus a row profile of the generated background (rows 0-80 from the top edge), all eyes.

Run from ~/BBDM (needs eval_volume_lq.py, eval_capdice_lq.py, capdice_patients.txt), CPU only:
    python bg_check.py --pids 10451 --out bg_test          # one eye (~2 min), check it runs
    python bg_check.py --workers 8 --out bg_check           # all 50 eyes
Outputs: per_eye.csv, summary.csv (medians), wilcoxon.csv (B vs A, same input), background_rows.png
"""
import argparse, csv, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_volume_lq import windows, stack                                   # noqa: E402
from eval_capdice_lq import NB, C, pred_index, crop_info, ilm_rows           # noqa: E402

ARMS = ["ROITOP", "LQAX2", "LQBX2", "LQCX2", "LQAX4", "LQBX4", "LQCX4"]
PAIRS = [("LQBX2", "LQAX2"), ("LQBX4", "LQAX4"), ("LQCX2", "LQAX2"), ("LQCX4", "LQAX4")]
TS = (0, 3, 5)
RETINA = 160
PROF = 80
_G = {}


def one_eye(pid):
    a = _G
    y0 = windows(pid, a["info_top"])
    ilm = np.stack([ilm_rows(pid, n) for n in range(1, NB + 1)])
    ilm = np.clip(np.nan_to_num(ilm, nan=np.nanmedian(ilm)) - y0[:, None], 0, C - 1)
    g = stack(pid, y0)
    rr = np.arange(C)[None, :, None]
    ret = (rr >= ilm[:, None, :]) & (rr < ilm[:, None, :] + RETINA)
    vit = rr < ilm[:, None, :] - 3
    top = np.zeros_like(ret); top[:, :20, :] = True
    bgm = vit & (g == 0)
    rows, prof = [], {}
    for arm in a["arms"]:
        p0 = stack(pid, y0, arm, a["files"][arm], a["info"][arm])
        prof[arm] = np.array([p0[:, i][bgm[:, i]].mean() * 255 if bgm[:, i].any() else np.nan for i in range(PROF)])
        for T in TS:
            p = np.where(p0 < T / 255.0, 0.0, p0) if T else p0
            sp, sr, sv, st = [], [], [], []
            for i in range(NB):
                s, m = structural_similarity(g[i], p[i], data_range=1.0, full=True)
                sp.append(s)
                sr.append(m[ret[i]].mean() if ret[i].any() else np.nan)
                sv.append(m[vit[i]].mean() if vit[i].any() else np.nan)
                st.append(m[top[i]].mean())
            r = {"patient": pid, "method": arm, "T": T, "ssim_patch": np.nanmedian(sp),
                 "ssim_retina": np.nanmedian(sr), "ssim_vitreous": np.nanmedian(sv), "ssim_top20": np.nanmedian(st)}
            if T == 0:
                r["bg_grey"] = float(p0[bgm].mean() * 255) if bgm.any() else np.nan
            else:
                r["real_dim_frac"] = float((g[ret] < T / 255.0).mean())
                r["gen_dim_frac"] = float((p0[ret] < T / 255.0).mean())
            rows.append(r)
    print(f"  done {pid}", flush=True)
    return rows, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--pids", nargs="+", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="bg_check")
    args = ap.parse_args()
    pids = args.pids or [int(x) for x in open("capdice_patients.txt").read().split()]
    files = {a: pred_index(a) for a in args.arms}
    arms = [a for a in args.arms if files[a]]
    print(f"runs: {arms}   eyes: {len(pids)}   thresholds (grey): {TS}")
    info = {a: crop_info(a) for a in arms}
    _G.update(arms=arms, files=files, info=info, info_top=info.get("ROITOP") or crop_info("ROITOP"))
    os.makedirs(args.out, exist_ok=True)
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            res = pool.map(one_eye, pids)
    else:
        res = [one_eye(p) for p in pids]
    rows = [r for rr, _ in res for r in rr]

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out, "per_eye.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    mets = ["ssim_patch", "ssim_retina", "ssim_vitreous", "ssim_top20", "bg_grey", "real_dim_frac", "gen_dim_frac"]
    summ = []
    for arm in arms:
        for T in TS:
            rs = [r for r in rows if r["method"] == arm and r["T"] == T]
            s = {"method": arm, "T": T, "n": len(rs)}
            for m in mets:
                v = [r[m] for r in rs if m in r and r[m] == r[m]]
                if v:
                    s[m] = round(float(np.median(v)), 4)
            summ.append(s)
    keys = list(dict.fromkeys(k for s in summ for k in s))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summ)

    wil = []
    for b, a in PAIRS:
        if b not in arms or a not in arms:
            continue
        for T in TS:
            for m in ("ssim_patch", "ssim_retina", "ssim_vitreous", "ssim_top20"):
                A = {r["patient"]: r[m] for r in rows if r["method"] == a and r["T"] == T}
                B = {r["patient"]: r[m] for r in rows if r["method"] == b and r["T"] == T}
                d = np.array([B[p] - A[p] for p in A if p in B and A[p] == A[p] and B[p] == B[p]])
                if len(d) < 5:
                    continue
                wil.append({"pair": f"{b}-{a}", "T": T, "metric": m, "median_delta": round(float(np.median(d)), 4),
                            "b_better": int((d > 0).sum()), "n": len(d),
                            "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
    with open(os.path.join(args.out, "wilcoxon.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wil[0])); w.writeheader(); w.writerows(wil)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for arm in arms:
        ax.plot(np.arange(PROF), np.nanmean([pr[arm] for _, pr in res], 0), label=arm, lw=1.3)
    ax.axhline(0, color="k", ls="--", lw=1, label="real OCTA (0)")
    ax.set_xlabel("row from the top edge of the (256×256) patch (px)")
    ax.set_ylabel("generated grey level (0-255)")
    ax.set_title(f"Generated background where the real OCTA is 0, mean over {len(pids)} test eyes", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "background_rows.png"), facecolor="white")

    print("\nmedian over eyes (T = generated pixels below T grey set to 0)")
    print(f"{'run':<8} {'T':>2} {'patch':>7} {'retina':>7} {'vitreous':>9} {'top20':>7} {'bg grey':>8} {'real<T':>7} {'gen<T':>7}")
    for s in summ:
        f = lambda k, fmt="{:7.3f}": fmt.format(s[k]) if k in s else "      -"
        print(f"{s['method']:<8} {s['T']:>2} {f('ssim_patch')} {f('ssim_retina')} {f('ssim_vitreous', '{:9.3f}')} "
              f"{f('ssim_top20')} {f('bg_grey', '{:8.2f}')} {f('real_dim_frac')} {f('gen_dim_frac')}")
    print("\nB/C vs A (same input), median per-eye difference")
    for r in wil:
        print(f"  {r['pair']:<14} T={r['T']}  {r['metric']:<14} {r['median_delta']:+.4f}  better {r['b_better']}/{r['n']}  p={r['p']:.2g}")
    print(f"\nSaved in {args.out}/")


if __name__ == "__main__":
    main()
