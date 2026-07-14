set -euo pipefail

FILE_ID="1Weldocl5FMnqkhx-Pw19rxKOeQIdzWKx"
OUT="jury_bundle.tar.gz"
CHECKSUM_FILE="scripts/jury_bundle.sha256"

if [ "$FILE_ID" = "REPLACE_WITH_DRIVE_FILE_ID" ]; then
  echo "!! scripts/fetch_jury_bundle.sh has no Drive file ID configured yet" >&2
  exit 1
fi

REQUIRED_GB=14
AVAIL_GB=$(($(df -Pk . | tail -1 | awk '{print $4}') / 1024 / 1024))
if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
  echo "!! Need ~${REQUIRED_GB} GiB free disk (bundle is ~6 GiB compressed + ~6 GiB extracted, coexisting mid-extract); only ${AVAIL_GB} GiB available in $(pwd)" >&2
  exit 1
fi

.venv/bin/python -m pip show gdown >/dev/null 2>&1 || .venv/bin/pip install --quiet gdown

echo ">> Downloading grading bundle (checkpoint + demo scenes + reference data + hand-labeled split, ~6 GiB)"
.venv/bin/gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUT"

if [ -f "$CHECKSUM_FILE" ]; then
  echo "$(cat "$CHECKSUM_FILE")  $OUT" | sha256sum -c -
fi

echo ">> Extracting into repo root"
tar -xzf "$OUT" -C .
rm -f "$OUT"

echo ">> Done. Try:"
echo "   make predict INPUT=data/demo_scenes/baku.tif OUTPUT=/tmp/baku_mask.tif"
echo "   make eval"
