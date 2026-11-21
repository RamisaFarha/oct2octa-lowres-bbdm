#!/usr/bin/env bash
# setup.sh - put the repository scripts where the pipeline expects them.
#   ~/lq2hq/scripts/   data, job and analysis scripts (linked, so a git pull updates them)
#   ~/lq2hq/config.sh  your paths (copied once from configs/config.example.sh; edit it)
#   $BBDM/             evaluation scripts and the list of test eyes (copied next to BBDM's main.py)
# Run once after cloning, and again after editing config.sh:  bash setup.sh
set -e
REPO=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HOME/lq2hq/scripts"
for f in "$REPO"/scripts/data/* "$REPO"/scripts/jobs/* "$REPO"/scripts/analysis/*; do
  ln -sf "$f" "$HOME/lq2hq/scripts/"
done
if [ ! -f "$HOME/lq2hq/config.sh" ]; then
  cp "$REPO/configs/config.example.sh" "$HOME/lq2hq/config.sh"
  echo "Created ~/lq2hq/config.sh - edit the paths in it, then run: bash setup.sh"
  exit 0
fi
source "$HOME/lq2hq/config.sh"
[ -d "$BBDM" ] || { echo "BBDM folder not found: $BBDM (see README, step 1)"; exit 1; }
cp "$REPO"/scripts/eval/*.py "$BBDM"/
cp "$REPO"/configs/capdice_patients.txt "$BBDM"/
for f in setup_p200.py setup_capdice.py; do
  [ -f "$BBDM/$f" ] || echo "WARNING: $BBDM/$f missing - copy it from prior_pipeline/ (see prior_pipeline/README.md)"
done
echo "Done. Scripts: ~/lq2hq/scripts   Evaluation: $BBDM   Next: README step 3"
