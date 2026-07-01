#!/bin/bash
# Reproducible AMD/ROCm environment setup for the retargeting pipeline.
#
# Why a script (not pyproject/conda): the ROCm stack can't be expressed as
# declarative pip deps — Warp + MuJoCo-Warp are source-built forks (with HIP
# arch flags), torch comes from a ROCm index, and the CUDA pins in
# env/retargeting.yml / pyproject.toml must be bypassed with --no-deps. See env/rocm.md.
#
# Prerequisites: a ROCm PyTorch environment (e.g. the container
#   rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1)
# where `python -c "import torch; print(torch.version.hip)"` prints a HIP version.
#
# Usage (from the retargeting/ directory):
#   WORK=/path/to/build/dir GFX_ARCH=gfx942 bash env/setup_rocm.sh
#   export PYTHONPATH="$WORK/warp:$WORK/mujoco_warp:$PYTHONPATH"   # (printed at the end)
set -euo pipefail

WORK="${WORK:-$PWD/.rocm_build}"
GFX_ARCH="${GFX_ARCH:-gfx942}"          # gfx94x / gfx95x only
WITH_GEOCALIB="${WITH_GEOCALIB:-1}"     # set 0 to skip the gravity model
# Repo `retargeting/` dir = parent of this script's env/ directory.
RETARGETING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$WORK"
echo "=== ROCm setup: WORK=$WORK  GFX_ARCH=$GFX_ARCH ==="
python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.is_available())"

# 1. Warp (ROCm fork). -O0 avoids a pathologically slow reduce.cu compile on gfx942.
if [ ! -d "$WORK/warp" ]; then
  git clone -b amd-integration https://github.com/ROCm/warp.git "$WORK/warp"
fi
( cd "$WORK/warp" && python build_lib.py --hip-arch="$GFX_ARCH" --hipcc-options="-O0" )

# 2. MuJoCo-Warp (ROCm fork; works with warp >= 1.13, not the pinned 5d5a645).
if [ ! -d "$WORK/mujoco_warp" ]; then
  git clone -b amd-integration https://github.com/ROCm/mujoco_warp.git "$WORK/mujoco_warp"
fi
export PYTHONPATH="$WORK/warp:$WORK/mujoco_warp:${PYTHONPATH:-}"
pip install -e "$WORK/mujoco_warp" --no-deps
python -c "import warp as wp; print('warp', wp.config.version)"   # -> 1.13.0+rocm.0

# 3. retargeting + pure-python runtime deps (skip the CUDA pins via --no-deps).
pip install -e "$RETARGETING_DIR" --no-deps
pip install scipy loguru tyro tqdm filelock omegaconf loop-rate-limiters \
  "imageio[ffmpeg]" opencv-python trimesh rtree shapely networkx coacd mink viser plotly "mujoco>=3.8.0"

# 4. (optional) GeoCalib — free gravity model for a fresh reconstruction dir.
if [ "$WITH_GEOCALIB" = "1" ]; then
  pip install "git+https://github.com/cvg/GeoCalib.git"
fi

cat <<EOF

=== ROCm setup complete ===
Add this to your shell before running the pipeline:

  export PYTHONPATH="$WORK/warp:$WORK/mujoco_warp:\$PYTHONPATH"

Then, from the retargeting/ directory:

  python launch.py --task whisking --raw-dir ../reconstruction/whisking \\
    --no-show-viewer --no-wait-on-finish

See env/rocm.md for details and current ROCm limitations.
EOF
