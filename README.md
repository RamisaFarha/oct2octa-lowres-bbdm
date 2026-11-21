# Low-resolution OCT → high-resolution OCTA with a latent Brownian bridge diffusion model

**ECGR 8119 Medical Artificial Intelligence, UNC Charlotte — final project (Fall 2026)**
Ramisa Farha · Advisor: Dr. Shahed Mohammed

OCT is cheap and widely available; OCTA, which shows retinal blood flow, needs more expensive hardware.
Generative models can translate OCT into OCTA, but every published model is trained and tested on
high-resolution scans. This project measures what happens when the **input OCT has lower resolution**
(×2 and ×4), and whether **fine-tuning the model on low-resolution input** recovers what is lost. Every
result is computed against the real OCTA of the same eye, on 50 held-out eyes of OCTA-500 (3 mm).

A step-by-step reproduction guide is in **[Reproduction_Guide.docx](Reproduction_Guide.docx)**.
This README is the short version of the same steps.

## Models

| Model | Short label | Training strategy | Run folders in the code |
|---|---|---|---|
| **FullRes-BBDM** | FullRes | Trained on full-resolution OCT (the baseline) | `ROITOP` (full-res input), `LQAX2`, `LQAX4` |
| **ResAdapt-BBDM** | ResAdapt-×2 / ResAdapt-×4 | FullRes-BBDM fine-tuned separately on ×2 or ×4 OCT (3 epochs) | `LQBX2`, `LQBX4`; cross-resolution `LQBX2X4`, `LQBX4X2` |
| **MultiRes-BBDM** | MultiRes | FullRes-BBDM fine-tuned on a mixture of full-resolution, ×2 and ×4 OCT (3 epochs) | `LQCHR`, `LQCX2`, `LQCX4` |

Some figures in `figures/` were made before the models were named and still say Model A / B / C:
A = FullRes-BBDM, B = ResAdapt-BBDM, C = MultiRes-BBDM.

---

## Main findings

| Run (model, input) | Large-vessel Dice ↑ | Capillary Dice ↑ | Vessel width error (µm) ↓ | 3D SSIM ↑ |
|---|---|---|---|---|
| **FullRes-BBDM, full-resolution input (upper reference)** | **0.625** | 0.289 | 0.40 | 0.452 |
| FullRes-BBDM, ×2 input | 0.600 | 0.307 | 0.78 | 0.446 |
| **ResAdapt-×2, ×2 input** | **0.613** | 0.300 | 0.71 | 0.471 |
| FullRes-BBDM, ×4 input | 0.537 | 0.293 | 0.68 | 0.431 |
| ResAdapt-×4, ×4 input | 0.537 | 0.276 | 0.65 | 0.467 |
| ResAdapt-×2, ×4 input (cross-resolution) | 0.544 | 0.288 | 0.72 | 0.460 |
| Chance: random-noise image | 0.003 | 0.271 | 0.49 | – |
| Chance: another eye's OCTA | 0.105 | 0.204 | 0.54 | – |

Medians over 50 test eyes, en-face slab image (304×256). Full table: `results/summary_medians.csv`.

1. **Graceful breakdown.** Large-vessel Dice keeps 96% of its full-resolution value at ×2 and 86% at ×4.
   The clearest damage is vessel **width** (calibre +8% at ×2), not vessel density.
2. **Resolution-adapted fine-tuning recovers real vessels.** ResAdapt-×2 beats FullRes-BBDM on large vessels in 32 of 50 eyes
   (median +0.004, Wilcoxon p = 0.025) and closes ≈ 26% of the per-eye gap to full resolution. Overlays show
   the gain comes from vessels FullRes-BBDM missed, not from fabricated ones. Capillary Dice gets worse in 48 of 50 eyes.
3. **SSIM is partly measuring background.** Generated OCTA has a faint haze (≈ 2 grey levels of 255) where
   the real OCTA is 0, strongest at the top of each (256×256) patch. Setting generated pixels below 5 grey to
   0 raises top-row SSIM from 0.56 to 0.97; about two thirds of ResAdapt-×2's SSIM lead over FullRes-BBDM was this background.
4. **Continuity is a trust signal.** Real vessels continue into neighbouring B-scans 82% of the time,
   generated ones 61–69%; continuous generated vessel pixels are correct 84% of the time, isolated ones 66%.

---

## Repository layout

```
setup.sh                 links scripts into ~/lq2hq/scripts, copies evaluation scripts into the BBDM folder
configs/
  config.example.sh      all paths and settings (copy to ~/lq2hq/config.sh and edit)
  capdice_patients.txt   the 50 test eyes (10451–10500)
scripts/
  data/                  01_make_lq_data.py (×2/×4 datasets), lqlib.py (degradations),
                         05_make_noise.py + noisemodel.py (optional noise study), 06_make_25d.py (2.5D inputs)
  jobs/                  02_make_train_configs.py, 03_make_jobs.py, make_test_cfg.py,
                         run_all.sh / worker.sh / status.sh / stop_all.sh (GPU job queue), 00_check.sh, speed_test.sh
  eval/                  eval_volume_lq.py (main metrics), eval_capdice_lq.py (shared loading + vessel code),
                         bscan_analysis.py, vessel_continuity.py, case_study.py, bg_check.py,
                         noise_stats.py, noise_check.py
  analysis/              make_tables.py (every summary number), train_eye_fig.py (training-eye figure)
  figures/               scripts for the report figures
prior_pipeline/          README: the four files reused from the prior ROITOP pipeline (FullRes-BBDM)
results/                 per-eye and summary CSVs behind every number above
figures/                 report figures
docs/Reproduction_Guide.docx
```

---

## How to reproduce

**Hardware used:** Linux server, 3× NVIDIA GTX 1080 Ti (11 GB). Sampling takes about 7.7 s per (256×256)
patch per GPU (200 steps), so one test run (50 eyes × 304 B-scans = 15,200 patches) is ≈ 33 GPU-hours.
Evaluation runs on CPU.

### 1. Get the data and the model code
- **OCTA-500** (3 mm subset, OCT and OCTA volumes): M. Li et al., *Medical Image Analysis*, 2024, available on
  IEEE DataPort. Each 3 mm volume has 304 B-scans of (640×304) pixels.
- **BBDM** (B. Li et al., CVPR 2023): clone the official repository and install its requirements; download
  the pretrained VQGAN-f4 (`pretrained/vq-f4/model.ckpt`).
- **Prior ROITOP pipeline** (FullRes-BBDM): copy the four files listed in `prior_pipeline/README.md` into the BBDM
  folder, and train or obtain the FullRes-BBDM checkpoint (`latest_model_8.pth`).

### 2. Environment
```bash
conda create -n BBDM python=3.9 -y && conda activate BBDM
pip install -r <BBDM repo>/requirements.txt
pip install -r requirements.txt
```

### 3. Install this repository
```bash
git clone https://github.com/<user>/oct2octa-lowres-bbdm.git && cd oct2octa-lowres-bbdm
bash setup.sh                 # first run creates ~/lq2hq/config.sh
nano ~/lq2hq/config.sh        # set BBDM, PY, DATA, RES, ROITOP_*, GPUS, WORK
bash setup.sh                 # second run links scripts and copies evaluation code
source ~/lq2hq/config.sh
export OCTA500_RAW=/path/to/OCTA-500/OCTA_3mm    # raw volumes, used by the evaluation
bash ~/lq2hq/scripts/00_check.sh                 # checks paths, GPUs, disk
```

### 4. Build the low-resolution datasets (CPU, ≈ 4 GB)
Each OCT patch (256×256) is area-averaged to (128×128) or (64×64) and enlarged back with bicubic
interpolation. The OCTA target is never changed (symlinks).
```bash
$PY ~/lq2hq/scripts/01_make_lq_data.py --limit 20    # quick test: look at a few images
$PY ~/lq2hq/scripts/01_make_lq_data.py               # full: training sets + 7 test chunks
```
Expected input quality: ×2 ≈ 32.6 dB PSNR, ×4 ≈ 29.7 dB.

### 5. Retrain and test (GPU)
```bash
$PY ~/lq2hq/scripts/02_make_train_configs.py --force   # ResAdapt-×2, ResAdapt-×4, MultiRes: FullRes + 3 epochs
$PY ~/lq2hq/scripts/03_make_jobs.py                    # job list: 3 trainings + 49 test jobs
bash ~/lq2hq/scripts/run_all.sh                        # one worker per GPU in GPUS; safe to log out
bash ~/lq2hq/scripts/status.sh                         # progress at any time
```
Cross-resolution runs (ResAdapt-×2 on ×4 input, ResAdapt-×4 on ×2 input) are added to the same queue:
```bash
for i in $CHUNKS; do
  printf "LQBX2X4_%s\t-\ttest LQBX2X4 %s X4 LQX2\n" $i $i >> $WORK/jobs.tsv
  printf "LQBX4X2_%s\t-\ttest LQBX4X2 %s X2 LQX4\n" $i $i >> $WORK/jobs.tsv
done
bash ~/lq2hq/scripts/run_all.sh
```
ResAdapt-BBDM starts from the FullRes-BBDM checkpoint and trains 3 more epochs on ×2 (or ×4) input with every
other setting unchanged; MultiRes-BBDM does the same on a mix of full-resolution, ×2 and ×4 patches. Outputs land in `$RES/OCTA500_3mm_VOL_<run>_<chunk>_TEST/LBBDM-f4/sample_to_eval/200`.

### 6. Evaluate (CPU, from the BBDM folder)
```bash
cd $BBDM
python eval_volume_lq.py --out eval_volume_lq                    # main table: Dice, width, SSIM, PSNR, LPIPS, Wilcoxon
python eval_volume_lq.py --arms ROITOP LQBX2X4 LQBX4X2 --out eval_bcross   # after the cross-resolution jobs (below)
python vessel_continuity.py --workers 8 --out continuity        # continuity and precision
python case_study.py --out case_study                           # representative eyes, colour overlays
python bscan_analysis.py --workers 8 --out bscan_analysis       # B-scan SSIM, local SSIM maps
python bg_check.py --workers 8 --out bg_check                   # background-haze check
$PY ~/lq2hq/scripts/train_eye_fig.py --gpu 0 --bscans 100 152 200   # optional: training-eye figure (GPU, ~5 min)
```
Each script writes `per_patient.csv` (or `per_eye.csv`), `summary.csv` and, where relevant, `wilcoxon.csv`.

### 7. Figures
From the repository root, with the CSVs in `results/`:
```bash
python scripts/figures/delta_fig.py      # per-eye ΔDice histograms -> figures/delta_dice_hist.png
python scripts/figures/avsb_scatter.py   # ResAdapt − FullRes vs FullRes -> figures/avsb_vs_A_scatter.png
python scripts/analysis/make_tables.py   # all summary tables, gap closed, outlier and trend checks, heatmap
```

### 8. Check your numbers
Compare your `summary.csv` medians with `results/summary_medians.csv`. Diffusion sampling is stochastic,
so expect differences in the third decimal; the comparisons (B vs A, eyes improved) should hold.

---

## Notes
- The noise-study scripts (`05_make_noise.py`, `noisemodel.py`, `noise_stats.py`, `noise_check.py`,
  `run_noise_batch1.sh`) are included for the follow-on work; their results are not part of this report.
- Run names in the code are listed in the Models table above.
- Eye 10454 (vitreous opacities) is the worst eye for every model; excluding it changes medians by ≤ 0.006.

## References
- M. Li et al. OCTA-500: A retinal dataset for optical coherence tomography angiography study. *Medical Image Analysis*, 2024.
- B. Li, K. Xue, B. Liu, Y.-K. Lai. BBDM: Image-to-image translation with Brownian bridge diffusion models. *CVPR*, 2023.
- R. H. Badhon et al. Quantitative characterization of retinal features in translated OCTA. *Experimental Biology and Medicine*, 2024.
