#!/usr/bin/env bash
# E2 — Sen1Floods11 downloader (hand-labeled benchmark set).
#
# Pulls the Sentinel-1 SAR chips + hand water masks that the data contract
# (DATA_CONTRACT.md) is built against, plus the OFFICIAL train/val/test split
# CSVs (B5/S2). Public bucket — anonymous access, no GCP auth needed.
#
# Idempotent: re-running only fetches changed/missing files (gsutil rsync).
#
# Usage:  bash scripts/download_data.sh
# Needs:  gsutil  (pip install gsutil  — already in .venv)

set -euo pipefail

BUCKET="gs://sen1floods11/v1.1"
HAND="${BUCKET}/data/flood_events/HandLabeled"

DEST="data/processed/sen1floods11"
SPLIT_DEST="data/splits/official"

mkdir -p "${DEST}/S1Hand" "${DEST}/LabelHand" "${SPLIT_DEST}"

echo ">> S1Hand  (Sentinel-1 VV/VH, 446 chips, ~696 MiB)"
gsutil -m rsync -r "${HAND}/S1Hand"    "${DEST}/S1Hand"

echo ">> LabelHand  (hand water masks, 446 chips)"
gsutil -m rsync -r "${HAND}/LabelHand" "${DEST}/LabelHand"

echo ">> Official splits (flood_handlabeled)"
gsutil -m cp "${BUCKET}/splits/flood_handlabeled/flood_train_data.csv" "${SPLIT_DEST}/train.csv"
gsutil -m cp "${BUCKET}/splits/flood_handlabeled/flood_valid_data.csv" "${SPLIT_DEST}/val.csv"
gsutil -m cp "${BUCKET}/splits/flood_handlabeled/flood_test_data.csv"  "${SPLIT_DEST}/test.csv"

echo ">> Done. Counts:"
echo "   S1Hand:    $(find "${DEST}/S1Hand"    -name '*.tif' | wc -l) tif"
echo "   LabelHand: $(find "${DEST}/LabelHand" -name '*.tif' | wc -l) tif"
