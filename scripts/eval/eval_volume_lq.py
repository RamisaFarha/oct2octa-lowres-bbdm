#!/usr/bin/env python3
"""
eval_volume_lq.py - the baseline ROITOP eval_volume.py (6var evaluation) extended to the LQ OCT -> HQ OCTA runs.

Unchanged from eval_volume.py: stacking, ROITOP window, full/slab projections, Frangi vessel maps,
BVC/BVD/BVT/VPI, large vessels, capillaries, Dice, volume metrics. ROITOP therefore reproduces the
baseline 6var numbers. Additions:
  - arms = ROITOP + LQ runs (read via eval_capdice_lq: ROITOP crop geometry, both result folders);
    only complete arms are evaluated
  - chance baselines per projection: NOISE (smoothed random en-face) and OTHER_EYE (another
    patient's real en-face), scored with the same vessel analysis
  - capillary/vessel miss and fabrication rates (1 px), sharpness ratio, slow-axis stripe ratio,
    matched-blur control (ROITOP en-face blurred to each arm's sharpness)

Run from ~/BBDM (needs eval_capdice_lq.py next to it):
    python eval_volume_lq.py --progress
    python eval_volume_lq.py --out eval_volume_lq_output

--- original description ---
eval_volume.py - STACKED (volume-level) evaluation on full test eyes, plus en-face vessel metrics
computed with a Python port of the lab's Quantitative_OCTA MATLAB code (Badhon / Le / Alam).

Uses the full-volume predictions made by setup_capdice.py + launch_capdice.sh (all 304 B-scans per eye).

SAME SIZE FOR EVERY METHOD
  Every method's B-scan output is first put back in the original 640x304 geometry
  (FULL / 1x2 / 1x4 resized back, ROITOP pasted at its crop row), then the SAME window is taken from
  all of them and from the real OCTA: rows y0..y0+255 (ROITOP rule) x columns 24..279 of each B-scan.
    stacked volume  : 304 B-scans x 256 depth x 256 columns  (identical for all methods and GT)
    en-face images  : 304 x 256 pixels, 9.87 um/pixel (3 mm / 304), identical for all methods and GT

1. Volume metrics (on the stacked 304x256x256 volume): PSNR, MAE, 3D SSIM (7x7x7 window),
   fine-detail ratio (high-frequency energy vs GT), speckle ratio.
2. En-face images, two projections (min-max normalised like folderwise_projection_ddbm.py):
     "full": mean over all 256 rows of the window (the whole retina, as in Badhon's projection code)
     "slab": mean over an ILM-anchored slab whose depth is matched to the OCTA-500 OCTA(ILM_OPL) maps
   En-face PSNR, SSIM, LPIPS (if installed).
3. Vessel analysis on each en-face image, identical for GT and every method (port of
   main_BVC_VPI_BVT_v3.m and main_BVD_6mm_dl_v2.m):
     vessel map : imadjust (1-99%) -> Frangi (sigma 1, beta 0.5, c 15, bright vessels) -> x1e5 ->
                  Otsu binarize (imbinarize) -> remove objects < 10 px (bwareaopen)
     BVC (um)   : pixel size x vessel pixels / skeleton pixels   (skeleton + 1 spur pass)
     VPI (%)    : vessel-boundary pixels / all pixels x 100
     BVT        : mean (geodesic / Euclidean length) over skeleton branches
     BVD (%)    : fractal-dimension vessel zone (FD 0.7-1, threshold mean+40) over the image
     large vessels = opening(vessel map, disk 2), >= 60 px ; capillaries = vessels - dilated large vessels
     Dice vs GT : vessel, large-vessel and capillary maps
4. Per-patient CSV, summary, paired Wilcoxon (ROITOP vs each method), QC panels.

Run from ~/BBDM after the capdice sampling has finished:
    python eval_volume.py --progress
    python eval_volume.py 2>&1 | tee eval_volume_output.txt
"""
import argparse, csv, glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from scipy.ndimage import (gaussian_filter, uniform_filter, binary_opening, binary_dilation,
                           binary_erosion, maximum_filter, minimum_filter, convolve, label)
from scipy.stats import wilcoxon
from skimage.filters import frangi, threshold_otsu
from skimage.metrics import structural_similarity
from skimage.morphology import disk, remove_small_objects, skeletonize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_capdice_lq import (RAW, PROJ, H, W, X0, C, NB, ARMS, NSPLIT, load01, pred_index, crop_info,
                             to_full, ilm_rows, ORIENT, corr, miss_fab, sharp, stripe, blur_to)
from setup_capdice import roitop_y0

PIX_UM = 3000.0 / 304          # en-face pixel size (um): 3 mm over 304 A-scans / B-scans
METRICS_VOL = ["psnr3d", "mae3d", "ssim3d", "hf_ratio3d", "speckle_ratio3d"]
VESSEL = ["BVD", "BVC", "VPI", "BVT"]


# ------------------------------------------------------------------ volumes
def windows(pid, info_top):
    y0 = []
    for n in range(1, NB + 1):
        nm = f"{pid}_{n:03d}"
        if nm in info_top:
            y0.append(info_top[nm])
        else:
            a = np.asarray(Image.open(f"{RAW}/OCT/{pid}/{n}.bmp").convert("L"))
            v = roitop_y0(a)
            y0.append(v if v is not None else (H - C) // 2)
    return np.array(y0)


def stack(pid, y0, arm=None, files=None, info=None):
    out = np.zeros((NB, C, C), np.float32)
    miss = 0
    for i, n in enumerate(range(1, NB + 1)):
        if arm is None:
            full = load01(f"{RAW}/OCTA/{pid}/{n}.bmp")
        else:
            full = to_full(arm, files, info, f"{pid}_{n:03d}")
            if full is None:
                miss += 1
                continue
        out[i] = full[y0[i]:y0[i] + C, X0:X0 + C]
    if miss:
        print(f"  WARNING {arm} {pid}: {miss} B-scans missing (left as zeros)")
    return out


def ssim3d(a, b, win=7):
    a = a.astype(np.float32); b = b.astype(np.float32)
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    saa = uniform_filter(a * a, win) - mu_a ** 2
    sbb = uniform_filter(b * b, win) - mu_b ** 2
    sab = uniform_filter(a * b, win) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(m.mean())


def vol_metrics(p, g):
    mse = float(np.mean((p - g) ** 2))
    hp = p - gaussian_filter(p, (0, 1.5, 1.5)); hg = g - gaussian_filter(g, (0, 1.5, 1.5))
    eps = 1e-8
    return {"psnr3d": 10 * np.log10(1 / mse) if mse > 0 else float("inf"),
            "mae3d": float(np.mean(np.abs(p - g))),
            "ssim3d": ssim3d(p, g),
            "hf_ratio3d": float(np.mean(hp ** 2) / (np.mean(hg ** 2) + eps)),
            "speckle_ratio3d": float((p.std() / (p.mean() + eps)) / (g.std() / (g.mean() + eps) + eps))}


# ------------------------------------------------------------------ en-face
def minmax(e):
    e = e - e.min()
    return e / e.max() if e.max() > 0 else e


def enface_full(v):
    return minmax(v.mean(axis=1))


def enface_slab(v, ilm_roi, depth):
    rows = np.arange(C)[None, :, None]
    m = (rows >= ilm_roi[:, None, :]) & (rows < ilm_roi[:, None, :] + depth)
    s = (v * m).sum(1) / np.maximum(m.sum(1), 1)
    return minmax(s)


# ------------------------------------------------------------------ vessel metrics (port of the MATLAB code)
def imadjust(e):
    lo, hi = np.percentile(e, 1), np.percentile(e, 99)
    return np.clip((e - lo) / (hi - lo + 1e-12), 0, 1)


def vessel_map(e):
    v = frangi(imadjust(e), sigmas=[1], beta=0.5, gamma=15, black_ridges=False) * 1e5
    u8 = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    t = threshold_otsu(u8) / 255.0 if u8.min() != u8.max() else 0.5
    return remove_small_objects(v > t, 10, connectivity=2)


N8 = np.ones((3, 3)); N8[1, 1] = 0


def skeleton(vm):
    sk = skeletonize(vm)
    nb = convolve(sk.astype(int), N8, mode="constant")
    return sk & ~((nb == 1) & sk)             # one 'spur' pass: drop end pixels


def bvt(sk):
    nb = convolve(sk.astype(int), N8, mode="constant")
    bp = sk & (nb >= 3)
    br = sk & ~binary_dilation(bp, disk(2))
    br = remove_small_objects(br, 3, connectivity=2)
    lab, n = label(br, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.nonzero(m)
        nbm = convolve(m.astype(int), N8, mode="constant")
        ends = np.argwhere(m & (nbm == 1))
        if len(ends) < 2:
            continue
        # path length: orthogonal links 1, diagonal links sqrt(2) (quasi-euclidean geodesic)
        ortho = (m[:, 1:] & m[:, :-1]).sum() + (m[1:, :] & m[:-1, :]).sum()
        diag = (m[1:, 1:] & m[:-1, :-1]).sum() + (m[1:, :-1] & m[:-1, 1:]).sum()
        length = ortho + np.sqrt(2) * diag
        eu = np.hypot(*(ends[0] - ends[1]))
        if eu > 0:
            out.append(length / eu)
    return float(np.mean(out)) if out else float("nan")


def bvd(e, thr_add=40, lo=0.7, hi=1.0):
    I = e * 255.0
    I = np.where(I < I.mean() + thr_add, 0, I)
    rs = [3, 5, 7, 9, 11]
    B = {}
    for r in rs:
        F = np.floor((maximum_filter(I, r, mode="constant") - minimum_filter(I, r, mode="constant")) / r) + 1
        B[r] = np.log(F * (36.0 / r ** 2))
    i = np.log(np.array(rs, float))
    nxx = np.dot(i, i) - i.sum() ** 2 / 5
    fd_stack = np.stack([B[11], B[9], B[7], B[5], B[3]])   # same (reversed) pairing as the MATLAB code
    nxy = np.tensordot(i, fd_stack, 1) - i.sum() * fd_stack.sum(0) / 5
    FD = nxy / nxx
    FD[np.isnan(FD)] = np.nanmin(FD)
    FD = minmax(FD)
    FD[(FD > 0.7624) & (FD < 0.7626)] = 0
    FD[(FD > 0.7633) & (FD < 0.7640)] = 0
    ves = ((FD >= lo) & (FD <= hi)) | (FD == 0.7619)
    return 100.0 * ves.mean()


def large_vessels(e, min_px=100):
    from skimage.filters import apply_hysteresis_threshold
    f = frangi(imadjust(e), sigmas=[2, 3, 4], beta=0.5, black_ridges=False)
    m = apply_hysteresis_threshold(f, np.percentile(f, 85), np.percentile(f, 93))
    return remove_small_objects(m, min_px, connectivity=2)


def vessel_features(e):
    vm = vessel_map(e)
    sk = skeleton(vm)
    large = large_vessels(e)
    cap = vm & ~binary_dilation(large, disk(2))
    perim = vm & ~binary_erosion(vm, np.ones((3, 3)))
    f = {"BVD": bvd(e),
         "BVC": PIX_UM * vm.sum() / max(sk.sum(), 1),
         "VPI": 100.0 * perim.mean(),
         "BVT": bvt(sk)}
    return f, vm, large, cap


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def enface_scores(e, g, lp):
    out = {"psnr": float(10 * np.log10(1 / max(np.mean((e - g) ** 2), 1e-12))),
           "ssim": float(structural_similarity(g, e, data_range=1.0))}
    if lp is not None:
        out["lpips"] = lp(e, g)
    return out


def save_panel(path, tiles, labels):
    gap = 6
    h, w = tiles[0].shape
    canvas = np.ones((h + 14, len(tiles) * (w + gap)))
    for i, t in enumerate(tiles):
        canvas[14:, i * (w + gap):i * (w + gap) + w] = t
    im = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    for i, lab in enumerate(labels):
        d.text((i * (w + gap) + 3, 1), lab, fill=(0, 0, 0))
    im.save(path)


def get_lpips():
    try:
        import lpips, torch
        net = lpips.LPIPS(net="alex", verbose=False).eval()
        def f(a, b):
            ta = torch.from_numpy(a).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
            tb = torch.from_numpy(b).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
            with torch.no_grad():
                return float(net(ta, tb).item())
        return f
    except Exception:
        return None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--ref", default="ROITOP")
    ap.add_argument("--out", default="eval_volume_lq_output")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--pids_file", default="capdice_patients.txt", help="eyes to evaluate (one id per line)")
    args = ap.parse_args()
    pids = [int(x) for x in open(args.pids_file).read().split()]
    files = {a: pred_index(a) for a in args.arms}
    for a in args.arms:
        print(f"[{a}] {len(files[a])}/{len(pids) * NB * NSPLIT.get(a, 1)} predictions")
    if args.progress:
        return
    have = lambda a: all(f"{p}_{n:03d}" in files[a] for p in pids for n in range(1, NB + 1))
    arms = [a for a in args.arms if files[a] and have(a)]
    skipped = [a for a in args.arms if files[a] and a not in arms]
    if skipped:
        print(f"skipping incomplete arms: {skipped}")
    if "ROITOP" not in arms:
        sys.exit("ROITOP predictions incomplete: needed as the reference")
    arms = ["ROITOP"] + [a for a in arms if a != "ROITOP"]
    BASE = ["NOISE", "OTHER_EYE"]
    print(f"evaluating: {arms} + chance baselines {BASE}")
    os.makedirs(os.path.join(args.out, "qc"), exist_ok=True)
    info = {a: crop_info(a) for a in arms}
    info_top = info.get("ROITOP") or crop_info("ROITOP")
    lp = get_lpips()

    # pass 1: ILM (in window coordinates) and slab depth matched to OCTA-500 ILM_OPL maps
    cache, scores = {}, {}
    for pid in pids:
        y0 = windows(pid, info_top)
        ilm = np.stack([ilm_rows(pid, n) for n in range(1, NB + 1)])
        ilm = np.nan_to_num(ilm, nan=np.nanmedian(ilm)) - y0[:, None]
        g = stack(pid, y0)
        cache[pid] = (y0, np.clip(ilm, 0, C - 1))
        ref = load01(f"{PROJ}/{pid}.bmp")
        for d in range(30, 241, 15):
            e = enface_slab(g, cache[pid][1], d)
            for on, of in ORIENT.items():
                r = of(ref)
                if r.shape == (NB, W):
                    scores.setdefault((d, on), []).append(corr(e, r[:, X0:X0 + C]))
        print(f"  pass 1: {pid}")
    (depth, on), cs = max(scores.items(), key=lambda kv: np.nanmean(kv[1]))
    c = float(np.nanmean(cs))
    print(f"\nslab matched to OCTA-500 ILM_OPL maps: {depth} px from the ILM, orientation {on}, r = {c:.3f}")

    def extra(e, eg, ves, cap, gves, gcap):
        cm, cf = miss_fab(cap, gcap)
        vm_, vf = miss_fab(ves, gves)
        return {"capillary_miss": cm, "capillary_fab": cf, "vessel_miss": vm_, "vessel_fab": vf,
                "sharpness_ratio": sharp(e) / (sharp(eg) + 1e-12),
                "stripe_ratio": stripe(e) / (stripe(eg) + 1e-12)}

    rows = []
    g_last = stack(pids[-1], *cache[pids[-1]][:1])
    prev_ge = {"full": enface_full(g_last),
               "slab": enface_slab(g_last, cache[pids[-1]][1], depth)}     # OTHER_EYE partner of eye 1
    for pid in pids:
        y0, ilm = cache[pid]
        g = stack(pid, y0)
        ge = {"full": enface_full(g), "slab": enface_slab(g, ilm, depth)}
        gf = {k: vessel_features(e) for k, e in ge.items()}
        panels = {k: ([e], [gf[k][1].astype(float)], [gf[k][3].astype(float)], ["GT"], [gf[k][2].astype(float)])
                  for k, e in ge.items()}
        for k in ge:
            row = {"patient": pid, "method": "GT", "projection": k, **gf[k][0]}
            rows.append(row)
        # ---- chance baselines, scored exactly like a method (no volume metrics)
        rng = np.random.default_rng(pid)
        for k in ge:
            gfeat, gves, glarge, gcap = gf[k]
            for name, e in (("NOISE", minmax(gaussian_filter(rng.random(ge[k].shape), 0.6))),
                            ("OTHER_EYE", prev_ge[k])):
                f, ves, large, cap = vessel_features(e)
                row = {"patient": pid, "method": name, "projection": k,
                       **{"enface_" + kk: vv for kk, vv in enface_scores(e, ge[k], lp).items()},
                       "vessel_dice": dice(ves, gves), "large_vessel_dice": dice(large, glarge),
                       "capillary_dice": dice(cap, gcap), **f, **extra(e, ge[k], ves, cap, gves, gcap)}
                for v in VESSEL:
                    row[v + "_abs_err"] = abs(f[v] - gfeat[v])
                rows.append(row)
        prev_ge = ge
        e_ref = {}
        for a in arms:
            p = stack(pid, y0, a, files[a], info[a])
            vm = vol_metrics(p, g)
            for k, fn in (("full", enface_full), ("slab", lambda v: enface_slab(v, ilm, depth))):
                e = fn(p)
                f, ves, large, cap = vessel_features(e)
                gfeat, gves, glarge, gcap = gf[k]
                row = {"patient": pid, "method": a, "projection": k, **vm,
                       **{"enface_" + kk: vv for kk, vv in enface_scores(e, ge[k], lp).items()},
                       "vessel_dice": dice(ves, gves), "large_vessel_dice": dice(large, glarge),
                       "capillary_dice": dice(cap, gcap), **f, **extra(e, ge[k], ves, cap, gves, gcap)}
                if a == "ROITOP":
                    e_ref[k] = e
                elif k in e_ref:   # matched-blur control: ROITOP en-face blurred to this arm's sharpness
                    bf, bves, blarge, bcap = vessel_features(minmax(blur_to(e_ref[k], sharp(e))))
                    row["blurref_capillary_dice"] = dice(bcap, gcap)
                    row["blurref_capillary_fab"] = miss_fab(bcap, gcap)[1]
                    row["blurref_vessel_dice"] = dice(bves, gves)
                for v in VESSEL:
                    row[v + "_abs_err"] = abs(f[v] - gfeat[v])
                rows.append(row)
                panels[k][0].append(e); panels[k][1].append(ves.astype(float))
                panels[k][2].append(cap.astype(float)); panels[k][3].append(a)
                panels[k][4].append(large.astype(float))
        for k, (en, vs, cp, lab, lg) in panels.items():
            save_panel(os.path.join(args.out, "qc", f"{pid}_{k}_large_vessels.png"), lg, lab)
            save_panel(os.path.join(args.out, "qc", f"{pid}_{k}_enface.png"), en, lab)
            save_panel(os.path.join(args.out, "qc", f"{pid}_{k}_vessels.png"), vs, lab)
            save_panel(os.path.join(args.out, "qc", f"{pid}_{k}_capillaries.png"), cp, lab)
        r = ORIENT[on](load01(f"{PROJ}/{pid}.bmp"))[:, X0:X0 + C]
        save_panel(os.path.join(args.out, "qc", f"{pid}_slab_check.png"), [minmax(r), ge["slab"]],
                   ["OCTA-500 ILM_OPL", "our slab, real OCTA"])
        print(f"  done {pid}")

    keys = sorted({k for r in rows for k in r} - {"patient", "method", "projection"},
                  key=lambda k: (k not in METRICS_VOL, k))
    with open(os.path.join(args.out, "per_patient.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "method", "projection"] + keys)
        w.writeheader(); w.writerows(rows)

    summ = []
    for proj in ("full", "slab"):
        print(f"\n=== {proj} en-face projection (volume metrics identical in both) ===")
        for a in ["GT"] + arms + BASE:
            rs = [r for r in rows if r["method"] == a and r["projection"] == proj]
            line = {"projection": proj, "method": a, "patients": len(rs)}
            for k in keys:
                v = np.array([r.get(k, np.nan) for r in rs], float)
                if np.all(np.isnan(v)):
                    continue
                line[k + "_mean"], line[k + "_std"] = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
            summ.append(line)
            print(f"{a:<8}" + "  ".join(f"{k}={line[k + '_mean']:.4f}" for k in keys if k + "_mean" in line))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        cols = sorted({k for s in summ for k in s}, key=lambda k: (k not in ("projection", "method", "patients"), k))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(summ)

    LOWER = {"mae3d", "enface_lpips", "capillary_fab", "capillary_miss", "vessel_fab", "vessel_miss"} | \
        {v + "_abs_err" for v in VESSEL}
    RATIO = {"hf_ratio3d", "speckle_ratio3d", "stripe_ratio", "sharpness_ratio"}
    test_keys = METRICS_VOL + ["enface_psnr", "enface_ssim", "enface_lpips", "vessel_dice",
                               "large_vessel_dice", "capillary_dice", "capillary_fab", "capillary_miss",
                               "stripe_ratio"] + [v + "_abs_err" for v in VESSEL]
    wr = []
    if args.ref in arms and len(pids) >= 6:
        for proj in ("full", "slab"):
            print(f"\nWilcoxon ({proj}), {args.ref} vs each method, paired over {len(pids)} eyes")
            ref = {r["patient"]: r for r in rows if r["method"] == args.ref and r["projection"] == proj}
            for a in arms + BASE:
                if a == args.ref:
                    continue
                oth = {r["patient"]: r for r in rows if r["method"] == a and r["projection"] == proj}
                block = []
                for k in test_keys:
                    if k not in ref[pids[0]] or k not in oth[pids[0]]:
                        continue
                    x = np.array([ref[p][k] for p in pids], float); y = np.array([oth[p][k] for p in pids], float)
                    if k in RATIO:
                        x, y = np.abs(np.log(x)), np.abs(np.log(y))
                    try:
                        pv = float(wilcoxon(x, y).pvalue)
                    except ValueError:
                        pv = float("nan")
                    better = x.mean() < y.mean() if (k in LOWER or k in RATIO) else x.mean() > y.mean()
                    block.append({"projection": proj, "other": a, "metric": k, "ref_mean": x.mean(),
                                  "other_mean": y.mean(), "p": pv, "ref_better": bool(better)})
                ps = sorted([(i, b["p"]) for i, b in enumerate(block) if b["p"] == b["p"]], key=lambda t: t[1])
                run = 0.0
                for j, (i, pv) in enumerate(ps):
                    run = max(run, min(1.0, (len(ps) - j) * pv)); block[i]["p_holm"] = run
                for b in block:
                    b.setdefault("p_holm", float("nan"))
                    print(f"  vs {a:<8}{b['metric']:<20}{args.ref} {b['ref_mean']:.4f}  {a} {b['other_mean']:.4f}  "
                          f"p={b['p']:.2g}  p_holm={b['p_holm']:.2g}  {'ROITOP better' if b['ref_better'] else 'other better'}")
                wr += block
        with open(os.path.join(args.out, "wilcoxon.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wr[0].keys()))
            w.writeheader(); w.writerows(wr)

    with open(os.path.join(args.out, "settings.txt"), "w") as f:
        f.write(f"patients: {pids}\n"
                f"stacked volume per eye and method: {NB} B-scans x {C} rows x {C} columns "
                f"(rows y0..y0+255 by the ROITOP rule, columns {X0}-{X0 + C - 1}); identical for all methods and GT\n"
                f"en-face size: {NB} x {C} px, {PIX_UM:.2f} um/px; min-max normalised\n"
                f"projections: full = mean over the 256-row window; slab = mean over {depth} px below the ILM "
                f"(matched to OCTA-500 OCTA(ILM_OPL), r = {c:.3f}, orientation {on})\n"
                "vessel map: imadjust 1-99%, Frangi sigma 1 beta 0.5 c 15 bright, x1e5, Otsu, remove <10 px\n"
                "BVC = pixel size x vessel px / skeleton px; VPI = boundary px %; BVT = mean geodesic/euclidean per branch;\n"
                "BVD = FD vessel zone 0.7-1 (threshold mean+40) % of image; large = opening disk2 >=60px; "
                "large = Frangi sigma 2-4, hysteresis 85/93 percentile, >=100 px; capillary = vessel - dilate(large,2)\n"
                f"arms: {arms}; chance baselines: NOISE (gaussian-smoothed random en-face, sigma 0.6), "
                "OTHER_EYE (previous patient's real en-face)\n"
                "miss/fab tolerance 1 px; stripe = slow/fast-axis mean abs difference, ratio to GT; "
                "matched-blur control = ROITOP en-face blurred to the arm's Laplacian-variance sharpness\n")
    print(f"\nSaved in {args.out}/")


if __name__ == "__main__":
    main()
