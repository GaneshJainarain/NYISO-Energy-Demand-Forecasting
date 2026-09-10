#!/usr/bin/env bash
# Build the Lambda deployment zip.
#
#   ./serving/build.sh              # arm64 (default)
#   ARCH=x86_64 ./serving/build.sh
#
# Installs Linux wheels regardless of the host OS, so this works from macOS.
# --no-deps so nothing unexpected is dragged in, but scipy IS required: xgboost
# imports it at module load, not lazily. The full set lands at ~158 MB unzipped
# / 46 MB zipped - inside both Lambda's 250 MB unzipped and 50 MB direct-upload
# limits, though not by a wide margin. Adding pandas would break the second.
set -euo pipefail

ARCH="${ARCH:-arm64}"
# manylinux_2_28, NOT manylinux2014: xgboost publishes its recent releases
# only under the newer tag, and the old tag silently resolves to 3.0.5 - which
# cannot read a model saved by 3.2.0.
case "$ARCH" in
  arm64)  PLAT=manylinux_2_28_aarch64 ;;
  x86_64) PLAT=manylinux_2_28_x86_64 ;;
  *) echo "unknown ARCH: $ARCH" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/serving/build"
ZIP="$ROOT/serving/lambda.zip"
PY="${PYTHON:-$ROOT/env/bin/python}"
PYVER="${PYVER:-3.13}"

# Pin xgboost to whatever trained the champion. Serving with a different
# version returns silently wrong numbers rather than failing.
XGB_VERSION="$("$PY" -c "import json;print(json.load(open('$ROOT/artifacts/metadata.json'))['xgboost_version'])" 2>/dev/null || true)"
if [ -z "$XGB_VERSION" ]; then
  echo "cannot read xgboost_version from artifacts/metadata.json - run scripts/export_model.py first" >&2
  exit 1
fi
echo "pinning xgboost==$XGB_VERSION (from artifacts/metadata.json)"

rm -rf "$BUILD" "$ZIP"
mkdir -p "$BUILD"

echo "installing wheels for $ARCH ($PLAT), python $PYVER..."
"$PY" -m pip install -q --target "$BUILD" \
  --platform "$PLAT" --implementation cp --python-version "$PYVER" \
  --only-binary=:all: --no-deps \
  "xgboost==$XGB_VERSION" numpy scipy holidays python-dateutil six

cp "$ROOT/serving/handler.py" "$BUILD/"

# Trim only bytecode. Two tempting removals are actually load-bearing:
#   */tests/     - numpy imports numpy._core.tests at load time
#   *.dist-info  - holidays reads its version via importlib.metadata
# Both were stripped in an earlier version of this script and both produced
# confusing runtime failures, so leave them alone.
find "$BUILD" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$BUILD" -name "*.pyc" -delete 2>/dev/null || true

echo "unzipped: $(du -sh "$BUILD" | cut -f1)"
(cd "$BUILD" && zip -qr "$ZIP" .)
echo "zip:      $(du -h "$ZIP" | cut -f1)  ->  serving/lambda.zip"
