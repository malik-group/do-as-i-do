# Retargeting on AMD GPUs (ROCm)

The default env (`env/retargeting.yml`, `pyproject.toml`) targets the CUDA stack:
`warp-lang==1.10.1`, `mujoco-warp @ …@5d5a645`, `cuda-toolkit`, and the `nvidia-*-cu13` wheels.
On AMD (gfx942 / MI300X-class) you swap Warp + MuJoCo-Warp for their official ROCm forks; the
retargeting code itself needs only the small graph-capture fallback in `retargeting/utils/mjwp.py`
(capture a CUDA graph when supported, else eager `mjwarp.step`).

**Verified on:** AMD Instinct MI300X / MI308X (gfx942), ROCm 7.2, container
`rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`.
Only `gfx94x` / `gfx95x` are in the ROCm/warp supported-arch list.

All commands below run from the `retargeting/` directory.

## Quick setup

From a ROCm PyTorch environment (see the container above), `env/setup_rocm.sh` automates the fork
builds and the `--no-deps` install:

```bash
WORK=/path/to/build GFX_ARCH=gfx942 bash env/setup_rocm.sh
export PYTHONPATH="/path/to/build/warp:/path/to/build/mujoco_warp:$PYTHONPATH"
```

The manual steps below document what the script does.

## 1. Build the ROCm forks of Warp + MuJoCo-Warp

```bash
# Warp (ROCm fork). -O0 avoids a pathologically slow reduce.cu compile on gfx942.
git clone -b amd-integration https://github.com/ROCm/warp.git
cd warp && python build_lib.py --hip-arch=gfx942 --hipcc-options="-O0" && cd ..

# MuJoCo-Warp (ROCm fork; works with warp >= 1.13, not the pinned 5d5a645).
git clone -b amd-integration https://github.com/ROCm/mujoco_warp.git

export PYTHONPATH="$PWD/warp:$PWD/mujoco_warp"
pip install -e mujoco_warp --no-deps
python -c "import warp as wp; print('warp', wp.config.version)"   # -> 1.13.0+rocm.0
```

## 2. Install retargeting without the CUDA pins

```bash
# --no-deps skips the pinned warp-lang==1.10.1 / mujoco==3.4.0 / mujoco-warp@5d5a645;
# the ROCm builds on PYTHONPATH are used instead.
pip install -e . --no-deps
pip install scipy loguru tyro tqdm filelock omegaconf loop-rate-limiters \
  "imageio[ffmpeg]" opencv-python trimesh rtree shapely networkx coacd mink viser plotly "mujoco>=3.8.0"
```

## 3. (Optional) gravity for a fresh reconstruction directory

Stage 1 needs `<raw_dir>/gravity.json`. The reconstruction gravity estimator uses GeoCalib
(free, Apache/MIT, weights auto-download, runs on ROCm torch):

```bash
pip install "git+https://github.com/cvg/GeoCalib.git"
python ../reconstruction/scripts/predict_video_gravity.py <frames_dir> \
  --device cuda:0 --output_path <raw_dir>/gravity.json
```

## 4. Run the pipeline

```bash
python launch.py --task whisking --raw-dir ../reconstruction/whisking \
  --no-show-viewer --no-wait-on-finish
```

On ROCm, `torch.cuda.is_available()` is `True` (HIP maps onto the CUDA API), so `device: cuda:0`
in the config resolves to the AMD GPU — no config change needed.

## Current ROCm limitations (throughput only, correctness unaffected)

- **CUDA graph conditional nodes are unsupported on HIP.** MuJoCo-Warp detects HIP and
  auto-disables `Model.opt.graph_conditional`, running the implicit-solver iteration loop as a
  host-side Python for-loop instead of an in-graph `while`. The physics are identical; the solver
  just loses in-graph early-exit and adds host↔device round-trips. Observed ~83 s/it at full config
  (`num_samples=1024`, `max_num_iterations=32`) on gfx942, GPU ~44% utilized.
- `utils/mjwp.py` likewise falls back to eager `mjwarp.step` if `wp.ScopedCapture` is unavailable.

## End-to-end validation

Full pipeline (stages 1–5, default/full config) on the shipped `whisking` demo:

- Final object tracking error: **pos = 0.0239 m, quat = 0.1193 rad (~6.8°)**, monotonic, no NaN.
- Output: `outputs/sharpa/right/whisking/0/trajectory_mjwp.npz`.
