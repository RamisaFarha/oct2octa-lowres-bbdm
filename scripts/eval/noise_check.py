#!/usr/bin/env python3
"""
noise_check.py - Step 3 check: does the calibrated noise model (noisemodel.py) produce images whose statistics
match real OCT, and does each level hit its target?

For a few test eyes, the raw B-scans (crop columns 24-279, (640x256)) are degraded with every noise level and
measured with exactly the same code as the real data (noise_stats.bscan_stats). Expected:
  clean   : reproduces the real values (floor grey 69, looks ~13.5, grains 2.7 / 1.6 x 1.3 px, SNR ~9.1 dB)
  NL<L>   : effective looks of the floor drop to L; grains stay real; floor level and SNR roughly unchanged
  NA<D>   : looks and grains unchanged; SNR (retina above floor) drops by about the expected amount
Each level gets PASS / CHECK flags. Also reports the PSNR of the degraded crop vs the clean crop (only for
comparison with the old SP1 / SP2 levels, 31.1 / 24.5 dB).

Run from ~/BBDM (needs noise_stats.py here and ~/lq2hq/scripts/noisemodel.py + lqlib.py):
    python noise_check.py --out noise_check                      # 5 eyes, every 8th B-scan, ~5-10 min, CPU
    python noise_check.py --eyes 10451 --step 16 --out nc_quick  # quick try
Outputs: check.csv (one row per level), per_eye.csv, fig/levels_<eye>.png
"""
import argparse, csv, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("LQ2HQ_SCRIPTS", os.path.expanduser("~/lq2hq/scripts")))
import noise_stats as ns          # noqa: E402
import noisemodel as nm           # noqa: E402
from lqlib import degrade         # noqa: E402

REAL = {"vit_med": 69.0, "looks": 13.9, "grain_ax": 2.75, "grain_lat": 2.66, "grain_vit_ax": 1.61,
        "grain_vit_lat": 1.31, "snr_db": 9.11}
GPD = nm.GREY_PER_DB


def make_transform(cond):
    if cond == "clean":
        return None

    def t(img, pid, n):
        return degrade(np.clip(np.round(img), 0, 255).astype(np.uint8), cond, f"{pid}_{n:03d}").astype(np.float64)
    return t


def eye_level(pid, cond, step):
    rows, keep = [], None
    tr = make_transform(cond)
    for n in range(1, ns.NB + 1, step):
        r = ns.bscan_stats(pid, n, tr)
        if r is None:
            continue
        rows.append(r[0])
        if keep is None and n >= ns.NB // 2:
            keep = r[1][0]
    med = lambda k: float(np.nanmedian([r[k] for r in rows]))
    e = {"patient": pid, "level": cond}
    for k in ("vit_med", "vit_std", "vit_skew", "vit_clip_frac", "ret_med", "grain_ax", "grain_lat",
              "grain_vit_ax", "grain_vit_lat", "snr_grey"):
        e[k] = med(k)
    e["looks_vit"] = ns.looks_from_std(e["vit_std"], GPD)        # variance-based: valid for added noise sources
    e["snr_db"] = e["snr_grey"] / GPD
    return e, keep


def expected(cond):
    if cond in nm.LOOKS_LEVELS:
        return nm.LOOKS_LEVELS[cond], REAL["snr_db"]
    if cond in nm.LOSS_LEVELS:
        a = 10 ** (-nm.LOSS_LEVELS[cond] / 10)
        s0 = 10 ** (REAL["snr_db"] / 10) - 1                    # retina signal above floor (linear)
        return REAL["looks"], 10 * np.log10(a * s0 + 1)
    return REAL["looks"], REAL["snr_db"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eyes", nargs="+", type=int, default=[10451, 10461, 10471, 10481, 10491])
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--levels", nargs="+", default=["clean"] + list(nm.CONDS))
    ap.add_argument("--out", default="noise_check")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "fig"), exist_ok=True)

    per_eye, crops = [], {}
    for cond in args.levels:
        for pid in args.eyes:
            e, crop = eye_level(pid, cond, args.step)
            per_eye.append(e)
            crops[(pid, cond)] = crop
        print(f"  measured {cond}", flush=True)

    # PSNR of the degraded crop vs clean crop (reference only)
    for e in per_eye:
        c, d = crops[(e["patient"], "clean")], crops[(e["patient"], e["level"])]
        mse = np.mean((c - d) ** 2)
        e["psnr_vs_clean"] = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)

    rows = []
    for cond in args.levels:
        es = [e for e in per_eye if e["level"] == cond]
        m = lambda k: float(np.nanmedian([e[k] for e in es]))
        L_exp, snr_exp = expected(cond)
        r = {"level": cond, "looks_target": L_exp, "looks_measured": m("looks_vit"),
             "snr_db_target": snr_exp, "snr_db_measured": m("snr_db"),
             "floor_grey": m("vit_med"), "floor_clip_frac": m("vit_clip_frac"),
             "grain_tissue": f"{m('grain_ax'):.2f} x {m('grain_lat'):.2f}",
             "grain_floor": f"{m('grain_vit_ax'):.2f} x {m('grain_vit_lat'):.2f}",
             "psnr_vs_clean": m("psnr_vs_clean")}
        ok_looks = abs(np.log(r["looks_measured"] / L_exp)) < np.log(1.3) if np.isfinite(r["looks_measured"]) else False
        ok_snr = abs(r["snr_db_measured"] - snr_exp) < 1.0
        ok_grain = (abs(m("grain_ax") - REAL["grain_ax"]) < 0.4 and abs(m("grain_vit_ax") - REAL["grain_vit_ax"]) < 0.4
                    and abs(m("grain_vit_lat") - REAL["grain_vit_lat"]) < 0.4)
        r["looks_ok"], r["snr_ok"], r["grain_ok"] = ok_looks, ok_snr, ok_grain
        r["clip_note"] = "floor clipped at 0: looks underestimated" if r["floor_clip_frac"] > 0.005 else ""
        rows.append(r)

    with open(os.path.join(args.out, "check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "per_eye.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_eye[0])); w.writeheader(); w.writerows(per_eye)

    print(f"\n{len(args.eyes)} eyes, every {args.step}th B-scan. Real data: looks 13.5, SNR 9.1 dB, "
          f"grain tissue 2.75 x 2.66, floor 1.61 x 1.31 px")
    print(f"{'level':<6} {'looks target':>12} {'measured':>9} {'SNR target':>11} {'measured':>9} {'floor':>6} "
          f"{'clip':>6} {'grain tissue':>13} {'grain floor':>12} {'PSNR':>6}  flags")
    for r in rows:
        flags = " ".join(f"{k[:-3]}:{'PASS' if r[k] else 'CHECK'}" for k in ("looks_ok", "snr_ok", "grain_ok"))
        print(f"{r['level']:<6} {r['looks_target']:12.1f} {r['looks_measured']:9.1f} {r['snr_db_target']:11.1f} "
              f"{r['snr_db_measured']:9.1f} {r['floor_grey']:6.1f} {100 * r['floor_clip_frac']:5.1f}% "
              f"{r['grain_tissue']:>13} {r['grain_floor']:>12} {r['psnr_vs_clean']:6.1f}  {flags} {r['clip_note']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pid = args.eyes[0]
    fig, ax = plt.subplots(1, len(args.levels), figsize=(2.3 * len(args.levels), 4.4), dpi=150)
    for j, cond in enumerate(args.levels):
        ax[j].imshow(crops[(pid, cond)][150:520], cmap="gray", vmin=0, vmax=255, aspect="auto")
        r = rows[j]
        ax[j].set_title(f"{cond}\nL {r['looks_measured']:.1f}, SNR {r['snr_db_measured']:.1f} dB\n(370×256)",
                        fontsize=8)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    fig.suptitle(f"Eye {pid}, central B-scan, rows 150-519 of the crop: calibrated noise levels "
                 "(NL = fewer averaged frames, NA = signal loss in dB)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(args.out, "fig", f"levels_{pid}.png"), facecolor="white")
    plt.close(fig)
    print(f"\nSaved in {args.out}/")


if __name__ == "__main__":
    main()
