#!/usr/bin/env python3
"""
eval_capdice_lq.py - eval_capdice.py extended to the LQ OCT -> HQ OCTA runs (course project).

Same pipeline, projection and segmentation as the baseline ROITOP eval_capdice.py, so ROITOP numbers reproduce the
baseline ones. Additions:
  - arms LQAX2 ... LQCX4 (and LQ*SP*): ROITOP crop geometry (same roi_log as ROITOP), outputs read from
    ~/BBDM/results and /data/BBDM_results
  - only arms whose predictions are complete are evaluated (no zero-filled volumes)
  - chance baselines NOISE (smoothed random noise) and OTHER_EYE (another patient's real OCTA),
    segmented and scored like a method: the Dice obtainable with no information about this eye
  - per patient: capillary/vessel miss and fabrication rates (1 px tolerance), sharpness ratio,
    slow-axis stripe ratio, and a matched-blur control (ROITOP-on-HR en-face blurred to the arm's
    sharpness, then segmented the same way)

Run from ~/BBDM:
    python eval_capdice_lq.py --progress
    python eval_capdice_lq.py                 # -> capdice_lq_output/

--- original description ---
eval_capdice.py - en-face capillary Dice for each method vs the real OCTA.

Pipeline (identical for the real OCTA and every method, so the comparison is fair):
  1. Rebuild each eye's OCTA volume from the method's predictions in original image coordinates
     (FULL / 1x2 / 1x4 resized back to 304x640, ROITOP pasted at its crop row, ROIFLAT un-flattened),
     columns 24-279 only (the region every method covers).
  2. En-face projection of a slab starting at the ILM (detected on the real OCT, same for all methods).
     Slab depth, mean/max projection and map orientation are chosen automatically so that the
     projection of the REAL volume best matches the OCTA-500 "OCTA(ILM_OPL)" map (sanity check).
  3. Vessel segmentation (local adaptive threshold), large vessels (morphological opening),
     capillaries = vessels minus dilated large vessels.
  4. Dice of each method's masks against the real OCTA's masks, per eye.

Run from ~/BBDM after the capdice sampling has finished:
    python eval_capdice.py --progress        # how many predictions exist per method
    python eval_capdice.py                   # full evaluation -> capdice_output/
"""
import argparse, csv, glob, os, sys, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_opening, binary_dilation, laplace
from scipy.stats import wilcoxon
from skimage.filters import threshold_local
from skimage.morphology import disk, remove_small_objects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup_p200 import detect_top_surface, clean_surface

RAW = os.environ.get("OCTA500_RAW", "/data/OCTA-500/OCTA_3mm")
PROJ = os.path.join(RAW, "Projection Maps", "OCTA(ILM_OPL)")
DROOT = os.environ.get("DATA", "/data/BBDM_data")
H, W, X0, C, NB = 640, 304, 24, 256, 304
ARMS = ["ROITOP", "LQAX2", "LQAX4", "LQBX2", "LQBX4", "LQCHR", "LQCX2", "LQCX4",
        "LQASP1", "LQASP2", "LQCSP1", "LQCSP2"]
RESULT_ROOTS = ["results", os.environ.get("RES", "/data/BBDM_results")]


def geom(arm):
    """LQ runs use the ROITOP crops (identical roi_log), so they are handled exactly like ROITOP."""
    return "ROITOP" if arm.startswith("LQ") else arm
NSPLIT = {"1x2": 2, "1x4": 4}


def load01(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0


def resize01(a, w, h):
    im = Image.fromarray((np.clip(a, 0, 1) * 255.0).astype(np.float32), mode="F")
    return np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float64) / 255.0


def pred_index(arm):
    """name -> png path, over all chunks of this arm."""
    out = {}
    dirs = []
    for r in RESULT_ROOTS:
        dirs += glob.glob(f"{r}/OCTA500_3mm_VOL_{arm}_*_TEST/LBBDM-f4/sample_to_eval/*")
    for d in dirs:
        if not os.path.basename(d).isdigit():
            continue
        for p in glob.glob(os.path.join(d, "*")):
            if p.lower().endswith((".png", ".bmp")):
                out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def crop_info(arm):
    arm = geom(arm)
    info = {}
    for root in glob.glob(f"{DROOT}/OCTA500_3mm_VOL_{arm}_*"):
        if arm == "ROITOP" and os.path.exists(f"{root}/roi_log.csv"):
            for r in csv.DictReader(open(f"{root}/roi_log.csv")):
                info[r["name"]] = int(r["roi_y0"])
        if arm == "ROIFLAT" and os.path.exists(f"{root}/shifts.npz"):
            info.update(dict(np.load(f"{root}/shifts.npz")))
    return info


def to_full(arm, files, info, name):
    """Prediction for one B-scan in original 640x304 coordinates (zeros where the method has no output)."""
    arm = geom(arm)
    if arm == "CENTER":                      # fixed centre crop, pasted back at rows 192..447
        p = files.get(name)
        if p is None:
            return None
        out = np.zeros((H, W))
        yc = (H - C) // 2
        out[yc:yc + C, X0:X0 + C] = load01(p)
        return out
    if arm in ("FULL", "ROITOP", "ROIFLAT"):
        p = files.get(name)
        if p is None:
            return None
        if arm == "FULL":
            return resize01(load01(p), W, H)
        out = np.zeros((H, W))
        if arm == "ROITOP":
            if name not in info:
                return None
            y0 = info[name]
            out[y0:y0 + C, X0:X0 + C] = load01(p)
            return out
        if name not in info:
            return None
        roi = load01(p)
        for i, s in enumerate(info[name]):
            lo = int(s)
            a, b = max(lo, 0), min(lo + C, H)
            if b > a:
                out[a:b, X0 + i] = roi[a - lo:b - lo, i]
        return out
    k = NSPLIT[arm]
    parts = []
    for i in range(k):
        p = files.get(f"{name}__p{i}")
        if p is None:
            return None
        parts.append(resize01(load01(p), W // k, H))
    return np.concatenate(parts, axis=1)


def gt_volume(pid):
    return np.stack([load01(f"{RAW}/OCTA/{pid}/{n}.bmp")[:, X0:X0 + C] for n in range(1, NB + 1)]).astype(np.float32)


def method_volume(arm, files, info, pid):
    sl, miss = [], 0
    for n in range(1, NB + 1):
        f = to_full(arm, files, info, f"{pid}_{n:03d}")
        if f is None:
            miss += 1
            f = np.zeros((H, W))
        sl.append(f[:, X0:X0 + C].astype(np.float32))
    if miss:
        print(f"  WARNING {arm} patient {pid}: {miss} B-scans missing (filled with zeros)")
    return np.stack(sl)


def ilm_rows(pid, n):
    a = np.asarray(Image.open(f"{RAW}/OCT/{pid}/{n}.bmp").convert("L"))
    ys = clean_surface(detect_top_surface(a))[X0:X0 + C]
    x = np.arange(C)
    ok = ~np.isnan(ys)
    if ok.sum() < 10:
        return np.full(C, np.nan)
    return np.interp(x, x[ok], ys[ok])


def project(slabs, ilm, depth, mode):
    """slabs: (304, 640, 256) volume; ilm: (304, 256). Returns (304, 256) en-face image."""
    rows = np.arange(H)[None, :, None]
    top = ilm[:, None, :]
    m = (rows >= top) & (rows < top + depth)
    v = np.where(m, slabs, np.nan)
    with np.errstate(all="ignore"):
        e = np.nanmax(v, axis=1) if mode == "max" else np.nanmean(v, axis=1)
    return np.nan_to_num(e)


ORIENT = {"id": lambda m: m, "T": lambda m: m.T, "fu": lambda m: m[::-1], "fl": lambda m: m[:, ::-1],
          "T_fu": lambda m: m.T[::-1], "T_fl": lambda m: m.T[:, ::-1], "r180": lambda m: m[::-1, ::-1],
          "T_r180": lambda m: m.T[::-1, ::-1]}


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def norm(e):
    lo, hi = np.percentile(e, 1), np.percentile(e, 99.5)
    return np.clip((e - lo) / (hi - lo + 1e-8), 0, 1)


def segment(e, block=31, offset=-0.02, min_vessel=10, large_r=2, min_large=60):
    p = gaussian_filter(norm(e), 0.6)
    vessel = p > threshold_local(p, block, method="gaussian", offset=offset)
    vessel &= p > 0.15
    vessel = remove_small_objects(vessel, min_vessel)
    large = remove_small_objects(binary_opening(vessel, disk(large_r)), min_large)
    cap = vessel & ~binary_dilation(large, disk(1))
    return vessel, large, cap


def miss_fab(gen, gt, tol=1):
    """fabricated = generated pixels with no real vessel within tol px; missed = the reverse."""
    gt_d, gen_d = binary_dilation(gt, disk(tol)), binary_dilation(gen, disk(tol))
    fab = float((gen & ~gt_d).sum() / gen.sum()) if gen.sum() else float("nan")
    miss = float((gt & ~gen_d).sum() / gt.sum()) if gt.sum() else float("nan")
    return miss, fab


def sharp(e):
    return float(laplace(norm(e)).var())


def stripe(e):
    """mean |row-to-row| change (rows = B-scans, slow axis) / mean |column-to-column| change."""
    n = norm(e)
    return float(np.abs(np.diff(n, axis=0)).mean() / (np.abs(np.diff(n, axis=1)).mean() + 1e-8))


def blur_to(e, target, iters=25):
    """Gaussian-blur e until its sharpness drops to target (bisection on sigma)."""
    if sharp(e) <= target:
        return e
    lo, hi = 0.0, 6.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sharp(gaussian_filter(e, mid)) > target else (lo, mid)
    return gaussian_filter(e, hi)


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def save_qc(path, tiles, labels):
    gap = 6
    h, w = tiles[0].shape
    canvas = np.ones((h + 14, len(tiles) * (w + gap)))
    for i, t in enumerate(tiles):
        canvas[14:, i * (w + gap):i * (w + gap) + w] = t
    im = Image.fromarray((canvas * 255).astype(np.uint8)).convert("RGB")
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    for i, lab in enumerate(labels):
        d.text((i * (w + gap) + 3, 1), lab, fill=(0, 0, 0))
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--ref", default="ROITOP")
    ap.add_argument("--out", default="capdice_lq_output")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--allow_partial", action="store_true", help="also evaluate incomplete arms (zero-filled)")
    args = ap.parse_args()
    pids = [int(x) for x in open("capdice_patients.txt").read().split()]
    files = {a: pred_index(a) for a in args.arms}
    need = {a: len(pids) * NB * NSPLIT.get(a, 1) for a in args.arms}
    for a in args.arms:
        print(f"[{a}] {len(files[a])}/{need[a]} predictions")
    if args.progress:
        return
    have = lambda a: all(f"{p}_{n:03d}" in files[a] for p in pids for n in range(1, NB + 1))
    arms = [a for a in args.arms if files[a] and (args.allow_partial or have(a))]
    skipped = [a for a in args.arms if a not in arms and files[a]]
    if skipped:
        print(f"skipping incomplete arms: {skipped}")
    if "ROITOP" not in arms:
        sys.exit("ROITOP predictions incomplete: needed as the reference")
    arms = ["ROITOP"] + [a for a in arms if a != "ROITOP"]
    print(f"evaluating: {arms}")
    os.makedirs(os.path.join(args.out, "qc"), exist_ok=True)
    info = {a: crop_info(a) for a in arms}

    # ---- pass 1: ILM per eye, and the projection that best matches OCTA-500's ILM_OPL map on the REAL data
    ilms, scores = {}, {}
    cands = [(d, m) for d in range(30, 241, 15) for m in ("mean", "max")]
    for pid in pids:
        ilm = np.stack([ilm_rows(pid, n) for n in range(1, NB + 1)])
        ilms[pid] = np.nan_to_num(ilm, nan=np.nanmedian(ilm))
        gt = gt_volume(pid)
        refmap = load01(f"{PROJ}/{pid}.bmp")
        for d, m in cands:
            e = project(gt, ilms[pid], d, m)
            for on, of in ORIENT.items():
                ref = of(refmap)
                if ref.shape == (NB, W):
                    scores.setdefault((d, m, on), []).append(corr(e, ref[:, X0:X0 + C]))
        print(f"  pass 1 done for {pid}")
    (depth, mode, on), cs = max(scores.items(), key=lambda kv: np.nanmean(kv[1]))
    c = float(np.nanmean(cs))
    print(f"\nprojection matched to OCTA-500 ILM_OPL maps: depth {depth} px, {mode} projection, "
          f"orientation {on}, mean correlation {c:.3f}")
    if c < 0.8:
        print("  NOTE: correlation below 0.8 - check capdice_output/qc/*_check.png before trusting results")

    rows = []
    prev_gt_e = None
    last_gt_e = project(gt_volume(pids[-1]), ilms[pids[-1]], depth, mode)   # partner for the first eye
    for pid in pids:
        gt_e = project(gt_volume(pid), ilms[pid], depth, mode)
        gv, gl, gc = segment(gt_e)
        ref = ORIENT[on](load01(f"{PROJ}/{pid}.bmp"))[:, X0:X0 + C]
        save_qc(os.path.join(args.out, "qc", f"{pid}_check.png"), [norm(ref), norm(gt_e)],
                ["OCTA-500 ILM_OPL", "our projection of real OCTA"])
        tiles, caps, labs = [norm(gt_e)], [gc.astype(float)], ["GT"]
        e_ref = None
        # ---- chance baselines (scored exactly like a method): what Dice does no information give?
        rng = np.random.default_rng(pid)
        noise = gaussian_filter(rng.random(gt_e.shape), 0.6)            # pure noise, slightly smoothed
        other = prev_gt_e if prev_gt_e is not None else last_gt_e       # another eye's real OCTA
        for name, e in (("NOISE", noise), ("OTHER_EYE", other)):
            v, l, cp = segment(e)
            cm, cf = miss_fab(cp, gc)
            vm, vf = miss_fab(v, gv)
            rows.append({"patient": pid, "method": name,
                         "capillary_dice": dice(cp, gc), "vessel_dice": dice(v, gv),
                         "large_vessel_dice": dice(l, gl),
                         "vessel_density_ratio": float(v.mean() / (gv.mean() + 1e-8)),
                         "capillary_density_ratio": float(cp.mean() / (gc.mean() + 1e-8)),
                         "enface_corr": corr(e, gt_e), "capillary_miss": cm, "capillary_fab": cf,
                         "vessel_miss": vm, "vessel_fab": vf,
                         "sharpness_ratio": sharp(e) / (sharp(gt_e) + 1e-12),
                         "stripe_ratio": stripe(e) / (stripe(gt_e) + 1e-12),
                         "blurref_capillary_fab": float("nan"), "blurref_capillary_dice": float("nan")})
        prev_gt_e = gt_e
        for a in arms:
            e = project(method_volume(a, files[a], info[a], pid), ilms[pid], depth, mode)
            v, l, cp = segment(e)
            cap_miss, cap_fab = miss_fab(cp, gc)
            ves_miss, ves_fab = miss_fab(v, gv)
            row = {"patient": pid, "method": a,
                   "capillary_dice": dice(cp, gc), "vessel_dice": dice(v, gv),
                   "large_vessel_dice": dice(l, gl),
                   "vessel_density_ratio": float(v.mean() / (gv.mean() + 1e-8)),
                   "capillary_density_ratio": float(cp.mean() / (gc.mean() + 1e-8)),
                   "enface_corr": corr(e, gt_e),
                   "capillary_miss": cap_miss, "capillary_fab": cap_fab,
                   "vessel_miss": ves_miss, "vessel_fab": ves_fab,
                   "sharpness_ratio": sharp(e) / (sharp(gt_e) + 1e-12),
                   "stripe_ratio": stripe(e) / (stripe(gt_e) + 1e-12),
                   "blurref_capillary_fab": float("nan"), "blurref_capillary_dice": float("nan")}
            if a == "ROITOP":
                e_ref = e
            elif e_ref is not None:   # matched-blur control: ROITOP-on-HR blurred to this arm's sharpness
                bv, bl, bc = segment(blur_to(e_ref, sharp(e)))
                row["blurref_capillary_fab"] = miss_fab(bc, gc)[1]
                row["blurref_capillary_dice"] = dice(bc, gc)
            rows.append(row)
            tiles.append(norm(e)); caps.append(cp.astype(float)); labs.append(a)
        save_qc(os.path.join(args.out, "qc", f"{pid}_enface.png"), tiles, labs)
        save_qc(os.path.join(args.out, "qc", f"{pid}_capillaries.png"), caps, labs)

    keys = ["capillary_dice", "vessel_dice", "large_vessel_dice", "vessel_density_ratio",
            "capillary_density_ratio", "enface_corr", "capillary_miss", "capillary_fab",
            "vessel_miss", "vessel_fab", "sharpness_ratio", "stripe_ratio",
            "blurref_capillary_fab", "blurref_capillary_dice"]
    with open(os.path.join(args.out, "per_patient.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "method"] + keys)
        w.writeheader(); w.writerows(rows)
    print(f"\nMean across {len(pids)} patients (density ratios: ideal 1)")
    print(f"{'method':<9}" + "".join(f"{k:>25}" for k in keys))
    summ = []
    for a in arms + ["NOISE", "OTHER_EYE"]:
        rs = [r for r in rows if r["method"] == a]
        line = {"method": a}
        for k in keys:
            v = np.array([r[k] for r in rs])
            line[k + "_mean"], line[k + "_std"] = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
        summ.append(line)
        print(f"{a:<9}" + "".join(f"{line[k + '_mean']:>18.4f}+-{line[k + '_std']:<5.3f}" for k in keys))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)

    if args.ref in arms and len(pids) >= 6:
        print(f"\nWilcoxon, {args.ref} vs each method (paired over patients)")
        ref = {r["patient"]: r for r in rows if r["method"] == args.ref}
        for a in arms + ["NOISE", "OTHER_EYE"]:
            if a == args.ref:
                continue
            oth = {r["patient"]: r for r in rows if r["method"] == a}
            for k in ("capillary_dice", "vessel_dice", "large_vessel_dice", "capillary_fab", "capillary_miss",
                      "stripe_ratio"):
                x = np.array([ref[p][k] for p in pids]); y = np.array([oth[p][k] for p in pids])
                try:
                    p = float(wilcoxon(x, y).pvalue)
                except ValueError:
                    p = float("nan")
                print(f"  vs {a:<8}{k:<20}{args.ref} {x.mean():.4f}  {a} {y.mean():.4f}  p={p:.2g}")
    with open(os.path.join(args.out, "settings.txt"), "w") as f:
        f.write(f"patients {pids}\nslab depth {depth} px from ILM, {mode} projection, orientation {on}, "
                f"corr with OCTA-500 ILM_OPL {c:.3f}\ncolumns {X0}-{X0 + C - 1}\n"
                "segmentation: gaussian 0.6, local threshold block 31 offset -0.02, p>0.15, min 10 px; "
                "large vessels: opening disk(2), min 60 px; capillaries = vessels - dilate(large, 1)\n"
                "miss/fab tolerance 1 px (disk); matched-blur control = ROITOP en-face blurred to each arm's "
                "Laplacian-variance sharpness\n")
    print(f"\nSaved in {args.out}/ (check qc/*_check.png, qc/*_enface.png, qc/*_capillaries.png)")


if __name__ == "__main__":
    main()
