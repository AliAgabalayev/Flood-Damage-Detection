set -euo pipefail

GSUTIL="$(command -v gsutil || echo .venv/bin/gsutil)"

BUCKET="gs://sen1floods11/v1.1"
HAND="${BUCKET}/data/flood_events/HandLabeled"

DEST="data/processed/sen1floods11"
SPLIT_DEST="data/splits/official"

mkdir -p "${DEST}/S1Hand" "${DEST}/LabelHand" "${SPLIT_DEST}"

echo ">> S1Hand  (Sentinel-1 VV/VH, 446 chips, ~696 MiB)"
"$GSUTIL" -m rsync -r "${HAND}/S1Hand"    "${DEST}/S1Hand"

echo ">> LabelHand  (hand water masks, 446 chips)"
"$GSUTIL" -m rsync -r "${HAND}/LabelHand" "${DEST}/LabelHand"

echo ">> Official splits (flood_handlabeled)"
"$GSUTIL" -m cp "${BUCKET}/splits/flood_handlabeled/flood_train_data.csv" "${SPLIT_DEST}/train.csv"
"$GSUTIL" -m cp "${BUCKET}/splits/flood_handlabeled/flood_valid_data.csv" "${SPLIT_DEST}/val.csv"
"$GSUTIL" -m cp "${BUCKET}/splits/flood_handlabeled/flood_test_data.csv"  "${SPLIT_DEST}/test.csv"

echo ">> Done. Counts:"
echo "   S1Hand:    $(find "${DEST}/S1Hand"    -name '*.tif' | wc -l) tif"
echo "   LabelHand: $(find "${DEST}/LabelHand" -name '*.tif' | wc -l) tif"
