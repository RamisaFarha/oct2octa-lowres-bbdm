# Copy to config.sh and edit the paths for your machine, then: source config.sh
#   source ~/lq2hq/config.sh

export BBDM=$HOME/BBDM                                   # BBDM code (main.py, configs/) with the ROITOP pipeline files
export PY=$HOME/miniconda3/envs/BBDM/bin/python
export DATA=/data/BBDM_data
export RES=/data/BBDM_results

# ROITOP (model A): training config, data and checkpoint of the baseline ROITOP model (Model A)
export ROITOP_TRAIN_CFG=$BBDM/configs/OCTA500_3mm_P200_ROITOP.yaml
export ROITOP_TRAIN_DATA=$DATA/OCTA500_3mm_P200_ROITOP256
export ROITOP_CKPT=$RES/OCTA500_3mm_P200_ROITOP/LBBDM-f4/checkpoint/latest_model_8.pth

# Test volumes: ROITOP chunks (50 eyes x 304 B-scans). Use fewer chunks to save GPU time:
#   0 2 4 = 30 eyes (10 each), 8 9 10 11 = 20 eyes (5 each)
export CHUNKS="0 2 4 8 9 10 11"

# Fine-tuning budget: extra epochs on top of ROITOP's 8 (set after checking epoch time)
export FT_EPOCHS=3

# Test batch size for the new runs (set from speed_test.sh; baseline used 1)
export TEST_BATCH=1

# GPUs to use (leave one free if you share the machine)
export GPUS="0 1 2"

# Bookkeeping for this project (small: logs, job markers, eval outputs)
export WORK=/data/lq2hq_work
