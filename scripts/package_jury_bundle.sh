set -euo pipefail

OUT="${OUT:-jury_bundle.tar.gz}"
CHECKSUM_FILE="scripts/jury_bundle.sha256"

PATHS=(
  models/segformer_b4/finetune/best.ckpt
  data/reference/dem
  data/reference/jrc_gsw
  data/demo_scenes
  data/processed/sen1floods11/S1Hand
  data/processed/sen1floods11/LabelHand
  data/splits/official
)

echo ">> Bundle contents:"
du -sh "${PATHS[@]}"

tar -czf "$OUT" "${PATHS[@]}"
sha256sum "$OUT" | awk '{print $1}' > "$CHECKSUM_FILE"

echo ">> Wrote $OUT ($(du -h "$OUT" | cut -f1)); checksum written to $CHECKSUM_FILE"
echo ">> Next: upload $OUT to the public jury Drive folder (Anyone with the link - Viewer),"
echo "   then set FILE_ID in scripts/fetch_jury_bundle.sh to its Drive file ID"
echo "   and 'git add $CHECKSUM_FILE && git commit'."
