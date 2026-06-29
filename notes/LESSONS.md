# Do as I Do — Lessons Learned

Paper: arXiv:2606.19333  
Date: 2026-06-29

---

## Environment

### CUDA version mismatch is not a blocker with pip-bundled CUDA
System CUDA 12.1, but `retargeting/pyproject.toml` pins `torch==2.12.1+cu130` (CUDA 13 via pip).
The pip-installed `nvidia-cuda-runtime` packages provide a self-contained CUDA runtime that does not conflict with the system CUDA installation. `torch.cuda.is_available()` returned `True` and the RTX 4070 was recognized.

### Warp kernel compilation: 75s first run, then cached
MuJoCo Warp JIT-compiles CUDA kernels on first use. Each kernel can take 1–25 seconds. Subsequent runs load from `~/.cache/warp/1.10.1/` in milliseconds. Always run a short test first (`--max-sim-steps 10`) to warm the cache before a long optimization run.

---

## Pipeline Architecture

### Four conda envs vs one
Reconstruction needs 4 separate environments (sam3, sam3d, hawor, tapnet) because each foundation model has incompatible CUDA/PyTorch pins. Retargeting uses a single env with pinned dependencies — much easier to install.

### `output_root_dir` inconsistency in launch.py
`launch.py` passes `output_root_dir` to `process_dataset` and `optimize_physics`, but not to `decompose_mesh`, `generate_scene`, `solve_ik`, or `resolve_scene_pedestal`. Those stages default to `f"{retargeting.ROOT}/../outputs"` which resolves to `retargeting/outputs/`. Workaround: do not pass `--output-root-dir` to `launch.py` — use the default `outputs/` dir. If you pass a custom dir, stages 2–4.5 will silently use `outputs/` anyway, then stage 5 will fail to find the scene it expects.

### max_sim_steps semantics: physics steps, warmup included in count
`max_sim_steps` counts MuJoCo physics steps (sim_dt=0.005s each), and warmup runs first. Since `warmup_steps=600`, you must pass `max_sim_steps > 600` to get any post-warmup optimization frames in the trajectory. With `max_sim_steps=300`, the NPZ stores 3 chunks but `replay_viser.py` skips 600 warmup frames and reports "No frames to play."

Minimum for a playable replay: `max_sim_steps >= 700` (warmup=600 + 1 ctrl step=100).

### config.json and gravity.json are generated, not committed
The reconstruction pipeline generates two metadata files in the video directory that are not committed to the repo:
- `config.json`: `{frame_number, object_names, anchor_hand}` — required by `process_dataset.py`
- `gravity.json`: `{vec3d, roll_deg, pitch_deg, n_frames, n_inliers, per_frame}` — required by `process_dataset.py`

When using pre-computed reconstruction outputs (as in this repo's whisking demo), create these manually. `vec3d` is the world-up direction in camera frame: for a level OpenCV-convention camera, `[0, -1, 0]`.

---

## VRAM Usage

### Full retargeting pipeline runs on 8 GB VRAM
With 1024 parallel environments, RTX 4070 Laptop (8 GB) handled the full pipeline:
- Stages 1–4: CPU only
- Stage 5 (MuJoCo Warp): peak GPU memory well within 8 GB (baseline 176 MiB idle, total peak confirmed by successful 9-minute 1000-step run without OOM)

### Reconstruction requires ≥32 GB VRAM — hard limit
Explicitly stated in `reconstruction/README.md`. Foundation models (SAM3, SAM3D, HaWoR) each require significant VRAM. Not feasible on a 8 GB laptop GPU.

---

## Repo State

### Fork was 4 days behind upstream at clone time
The upstream `malik-group/do-as-i-do` received key commits (whisking demo data, replay viewer) after the `curieuxjy/do-as-i-do` fork was created. Syncing from upstream is necessary to get the pre-computed demo files.

### Pre-computed whisking demo committed in upstream
The upstream repo ships:
- `reconstruction/whisking/whisking.mp4` — input video
- `reconstruction/whisking/whisking/all_hand_meshes.npz` — HaWoR hand reconstruction output
- `reconstruction/whisking/video_segmentation/masks/.../whisk.obj` — SAM3D mesh
- `reconstruction/whisking/obj_tracking_out/whisk/.../layout_camera_frame_optimized.json` — guided diffusion tracking output
- `retargeting/outputs/sharpa/right/whisking/0/trajectory_mjwp.npz` — full 1545-step optimization result

This enables testing the retargeting and replay pipeline without needing a GPU ≥32 GB.
