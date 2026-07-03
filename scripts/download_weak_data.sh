set -euo pipefail

GSUTIL="$(command -v gsutil || echo .venv/bin/gsutil)"

BUCKET="gs://sen1floods11/v1.1"
WEAK="${BUCKET}/data/flood_events/WeaklyLabeled"

DEST="data/processed/sen1floods11_weak"

mkdir -p "${DEST}/S1Weak" "${DEST}/S2IndexLabelWeak"

echo ">> S1Weak  (Sentinel-1 VV/VH, 4384 chips, ~6.8 GiB)"
"$GSUTIL" -m rsync -r "${WEAK}/S1Weak" "${DEST}/S1Weak"

echo ">> S2IndexLabelWeak  (Sentinel-2 water-index pseudo-labels, 4385 chips)"
"$GSUTIL" -m rsync -r "${WEAK}/S2IndexLabelWeak" "${DEST}/S2IndexLabelWeak"

echo ">> Done. Counts:"
echo "   S1Weak:           $(find "${DEST}/S1Weak"           -name '*.tif' | wc -l) tif"
echo "   S2IndexLabelWeak: $(find "${DEST}/S2IndexLabelWeak" -name '*.tif' | wc -l) tif"
