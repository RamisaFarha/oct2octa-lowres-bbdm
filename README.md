# Prior ROITOP pipeline (Model A)

This project builds on an existing latent-BBDM OCT-to-OCTA pipeline with retina-focused (ROITOP) cropping.
Model A is that pipeline's model, trained 8 epochs on full-resolution OCT. The scripts in this repository
reuse four of its files. Copy them here (and into your BBDM folder) before running anything:

| File | Used by | What it does |
|---|---|---|
| `setup_p200.py` | `eval_capdice_lq.py` | ILM (top retinal surface) detection: `detect_top_surface`, `clean_surface` |
| `setup_capdice.py` | `eval_volume_lq.py` | ROITOP window rule: `roitop_y0` (row where each (256×256) patch starts) |
| `configs/OCTA500_3mm_P200_ROITOP.yaml` | `02_make_train_configs.py` | Model A training config (copied for Models B and C) |
| `configs/OCTA500_3mm_VOL_ROITOP_<chunk>_TEST.yaml` | `make_test_cfg.py` | test configs for the 7 test chunks |

From the machine that has the pipeline:

```bash
cp ~/BBDM/setup_p200.py ~/BBDM/setup_capdice.py prior_pipeline/
mkdir -p prior_pipeline/configs
cp ~/BBDM/configs/OCTA500_3mm_P200_ROITOP.yaml ~/BBDM/configs/OCTA500_3mm_VOL_ROITOP_*_TEST.yaml prior_pipeline/configs/
```

The ROITOP crop: each B-scan of OCTA-500 3 mm is (640×304) pixels. Columns 24–279 are kept, and a 256-row
window is placed so it starts just above the retina (rule in `roitop_y0`), giving one (256×256) patch per
B-scan. Training data: ROITOP patches of the training eyes (`OCTA500_3mm_P200_ROITOP256`, folders
`train|val|test/A` = OCT and `/B` = OCTA). Test data: all 304 B-scans of the 50 test eyes, in 7 chunks
(`OCTA500_3mm_VOL_ROITOP_<chunk>`, with `roi_log.csv` giving each patch's window row).
