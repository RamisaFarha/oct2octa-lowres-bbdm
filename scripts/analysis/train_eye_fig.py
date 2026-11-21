#!/usr/bin/env python3
"""
train_eye_fig.py - redo the B-scan figure (OCT input | generated OCTA | real OCTA | local SSIM) for a TRAINING eye,
to see whether the dark-background / top-edge problem also happens on data the models were trained on.

Steps (all in one run, ~5-10 min on one GPU):
  1. picks B-scans of one training eye from the training data (ROITOP crop, (256x256) patches):
       HR input  = OCTA500_3mm_P200_ROITOP256/train/A        (what model A was trained on)
       x2 / x4   = OCTA500_3mm_P200_LQX2_256 / _LQX4_256 train/A (exactly what models B and C were trained on)
  2. builds tiny test folders  $DATA/OCTA500_3mm_TRFIG_<HR|X2|X4>  (links only, nothing is copied or changed)
  3. writes one test config per run (copy of the ROITOP chunk-0 test config; only dataset_name, dataset_path
     and model_load_path change) and samples with main.py --sample_to_eval, like the normal test jobs
  4. draws the figure, plus a second figure: brightness of the generated background (where the real OCTA is 0,
     above the retina) row by row from the top edge of the patch down

Runs drawn (same as the test-eye figure, plus B x2):
  A HR, A x2, A x4, B x2 (own input), B x4 (own input), C x4

Usage (from anywhere; check `nvidia-smi` first and pick a GPU with ~4 GB free):
  source ~/lq2hq/config.sh
  $PY ~/lq2hq/scripts/train_eye_fig.py --gpu 3                       # first training eye, B-scan nearest 152
  $PY ~/lq2hq/scripts/train_eye_fig.py --gpu 3 --eye 10301 --bscans 100 152 200
  $PY ~/lq2hq/scripts/train_eye_fig.py --skip_sampling               # only redraw (samples already made)
Output: ~/BBDM/train_eye_fig/<eye>_bscan_<n>.png, <eye>_background_rows.png, background.csv
"""
import argparse, csv, glob, os, re, subprocess, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter
from skimage.metrics import structural_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_test_cfg import checkpoint  # noqa: E402  (newest checkpoint of a training run, or ROITOP_CKPT)

BBDM = os.environ.get("BBDM", os.path.expanduser("~/BBDM"))
DATA = os.environ.get("DATA", "/data/BBDM_data")
RES = os.environ.get("RES", "/data/BBDM_results")
PY = os.environ.get("PY", sys.executable)
SRC = {"HR": os.environ.get("ROITOP_TRAIN_DATA", f"{DATA}/OCTA500_3mm_P200_ROITOP256"),
       "X2": f"{DATA}/OCTA500_3mm_P200_LQX2_256", "X4": f"{DATA}/OCTA500_3mm_P200_LQX4_256"}
# run -> (input condition, model, label)
RUNS = {"TRFIG_A_HR": ("HR", "ROITOP", "A, HR input"),
        "TRFIG_A_X2": ("X2", "ROITOP", "A, ×2 input"),
        "TRFIG_A_X4": ("X4", "ROITOP", "A, ×4 input"),
        "TRFIG_B_X2": ("X2", "LQX2", "B (ft ×2), ×2 input"),
        "TRFIG_B_X4": ("X4", "LQX4", "B (ft ×4), ×4 input"),
        "TRFIG_C_X4": ("X4", "LQMIX", "C (mixed), ×4 input")}
RETINA = 160
EXTS = (".bmp", ".png", ".jpg", ".tif", ".tiff")


def load01(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def pick_names(eye, bscans):
    files = sorted(f for f in os.listdir(f"{SRC['HR']}/train/A") if f.lower().endswith(EXTS))
    eyes = sorted({f.split("_")[0] for f in files})
    eye = str(eye or eyes[0])
    mine = [f for f in files if f.split("_")[0] == eye]
    if not mine:
        sys.exit(f"eye {eye} not in the training split; training eyes are {eyes[0]}..{eyes[-1]} ({len(eyes)} eyes)")
    num = {f: int(re.findall(r"\d+", os.path.splitext(f)[0])[-1]) for f in mine}
    out = []
    for b in bscans:
        f = min(mine, key=lambda f: abs(num[f] - b))
        if f not in out:
            out.append(f)
    print(f"training eye {eye}: {len(mine)} patches in train/A; using {', '.join(out)}")
    return eye, out


def build_data(names):
    for cond, src in SRC.items():
        dst = f"{DATA}/OCTA500_3mm_TRFIG_{cond}"
        for split in ("train", "val", "test"):          # BBDM builds train/val sets even in test mode
            for sub in ("A", "B"):
                d = f"{dst}/{split}/{sub}"
                os.makedirs(d, exist_ok=True)
                for old in os.listdir(d):
                    os.remove(os.path.join(d, old))
            for n in (names if split == "test" else names[:1]):
                link(f"{src}/train/A/{n}", f"{dst}/{split}/A/{n}")
                link(f"{SRC['HR']}/train/B/{n}", f"{dst}/{split}/B/{n}")   # OCTA target is never degraded
    print(f"test folders: {DATA}/OCTA500_3mm_TRFIG_{{HR,X2,X4}}")


def make_cfg(run, cond, model):
    base = f"{BBDM}/configs/OCTA500_3mm_VOL_ROITOP_0_TEST.yaml"
    t = open(base).read()
    for pat, rep in ((r"(^\s*dataset_name:\s*).*$", rf"\1'{run}'"),
                     (r"(^\s*dataset_path:\s*).*$", rf"\1'{DATA}/OCTA500_3mm_TRFIG_{cond}'"),
                     (r"(^\s*model_load_path:\s*).*$", rf"\1'{checkpoint(model)}'")):
        t, n = re.subn(pat, rep, t, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"pattern not found in {base}: {pat}")
    out = f"{BBDM}/configs/{run}_TEST.yaml"
    open(out, "w").write(t)
    return out


def out_dir(run):
    return f"{RES}/{run}/LBBDM-f4/sample_to_eval/200"


def sample(gpu):
    for run, (cond, model, _) in RUNS.items():
        cfg = make_cfg(run, cond, model)
        log = f"{BBDM}/train_eye_fig/{run}.log"
        print(f"sampling {run} ({model} on {cond}) ... log: {log}", flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with open(log, "w") as f:
            r = subprocess.run([PY, "-u", "main.py", "--config", cfg, "--sample_to_eval", "--gpu_ids", "0",
                                "--port", str(12490 + int(gpu)), "--result_path", RES],
                               cwd=BBDM, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"{run} failed, see {log}")


def find_out(run, name):
    stem = os.path.splitext(name)[0]
    hit = [p for p in glob.glob(f"{out_dir(run)}/{stem}.*") if p.lower().endswith(EXTS)]
    return load01(hit[0]) if hit else None


def ilm_from_oct(oct01):
    """top retinal surface per column of an HR OCT patch (rows; 0 if the retina already starts at the top edge)."""
    sm = gaussian_filter1d(oct01, 2, axis=0)
    vit = np.percentile(sm[:20], 20)
    thr = vit + 0.35 * (np.percentile(sm, 95) - vit)
    above = sm > thr
    ilm = np.where(above.any(0), above.argmax(0), 0).astype(float)
    return median_filter(ilm, size=15, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye", default=None, help="training eye ID (default: first training eye)")
    ap.add_argument("--bscans", nargs="+", type=int, default=[152])
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--skip_sampling", action="store_true")
    args = ap.parse_args()
    os.makedirs(f"{BBDM}/train_eye_fig", exist_ok=True)
    eye, names = pick_names(args.eye, args.bscans)
    if not args.skip_sampling:
        if args.gpu is None:
            sys.exit("give --gpu (check nvidia-smi for one with ~4 GB free memory)")
        build_data(names)
        sample(args.gpu)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_csv, prof = [], {r: [] for r in RUNS}
    for name in names:
        g = load01(f"{SRC['HR']}/train/B/{name}")
        hr = load01(f"{SRC['HR']}/train/A/{name}")
        ilm = ilm_from_oct(hr)
        rr = np.arange(g.shape[0])[:, None]
        mask = (rr >= ilm[None, :]) & (rr < ilm[None, :] + RETINA)
        bg = (rr < ilm[None, :] - 3) & (g == 0)              # vitreous where the real OCTA is exactly 0
        top = max(float(np.percentile(g, 99.5)), 1e-3)
        fig, ax = plt.subplots(len(RUNS), 4, figsize=(9.6, 2.5 * len(RUNS) + 0.6), dpi=150)
        for r, (run, (cond, model, lab)) in enumerate(RUNS.items()):
            p = find_out(run, name)
            if p is None:
                sys.exit(f"no output for {run} {name} in {out_dir(run)}")
            s, m = structural_similarity(g, p, data_range=1.0, full=True)
            inp = load01(f"{SRC[cond]}/train/A/{name}")
            for c, (im, cm, lo, hi) in enumerate([(inp, "gray", 0, 1), (p, "gray", 0, top), (g, "gray", 0, top),
                                                  (m, "magma", 0, 1)]):
                ax[r, c].imshow(im, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                for c, t in enumerate(["OCT input", "generated OCTA", "real OCTA", "local SSIM map"]):
                    ax[0, c].set_title(f"{t}\n(256×256)", fontsize=9)
            ax[r, 3].contour(mask, levels=[0.5], colors="#3ddc97", linewidths=0.8)
            top10 = bg[:10]
            bg_top = float(p[:10][top10].mean()) if top10.any() else float("nan")
            bg_rest = float(p[10:][bg[10:]].mean()) if bg[10:].any() else float("nan")
            ax[r, 3].text(0.98, 0.03, f"patch {s:.3f}\nretina {m[mask].mean():.3f}", transform=ax[r, 3].transAxes,
                          ha="right", va="bottom", fontsize=7.5, color="white",
                          bbox=dict(facecolor="black", alpha=0.55, pad=1.5, edgecolor="none"))
            ax[r, 1].text(0.02, 0.97, f"background: top 10 rows {255 * bg_top:.1f}, below {255 * bg_rest:.1f} grey",
                          transform=ax[r, 1].transAxes, ha="left", va="top", fontsize=6.5, color="#ffd166")
            ax[r, 0].set_ylabel(lab, fontsize=9)
            rows_csv.append({"eye": eye, "patch": name, "run": run, "ssim_patch": round(s, 4),
                             "ssim_retina": round(float(m[mask].mean()), 4),
                             "ssim_background": round(float(m[bg].mean()), 4) if bg.any() else "",
                             "bg_grey_top10": round(255 * bg_top, 2), "bg_grey_below": round(255 * bg_rest, 2)})
            pr = np.array([p[i][bg[i]].mean() if bg[i].any() else np.nan for i in range(80)])
            prof[run].append(pr)
        fig.suptitle(f"TRAINING eye {eye}, patch {os.path.splitext(name)[0]}: OCT input, generated and real OCTA, "
                     f"local SSIM (7×7; bright = similar).\nGreen = retina region (top surface to +{RETINA} px). "
                     f"Yellow = mean generated grey level where the real OCTA is 0 above the retina (real = 0).",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = f"{BBDM}/train_eye_fig/{eye}_bscan_{os.path.splitext(name)[0]}.png"
        fig.savefig(out, facecolor="white"); plt.close(fig)
        print("saved", out)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for run, (cond, model, lab) in RUNS.items():
        ax.plot(np.arange(80), 255 * np.nanmean(prof[run], 0), label=lab, lw=1.4)
    ax.axhline(0, color="k", lw=1, ls="--", label="real OCTA (0)")
    ax.set_xlabel("row from the top edge of the patch (px)"); ax.set_ylabel("generated grey level (0-255)")
    ax.set_title(f"Training eye {eye}: generated background where the real OCTA is 0 (above the retina)", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{BBDM}/train_eye_fig/{eye}_background_rows.png", facecolor="white")
    with open(f"{BBDM}/train_eye_fig/background.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)
    print(f"saved {BBDM}/train_eye_fig/{eye}_background_rows.png and background.csv")
    for r in rows_csv:
        print(f"  {r['run']:<11} {r['patch']:<14} SSIM patch {r['ssim_patch']:.3f} retina {r['ssim_retina']:.3f} "
              f"background {r['ssim_background']}  bg grey top10 {r['bg_grey_top10']} below {r['bg_grey_below']}")


if __name__ == "__main__":
    main()
