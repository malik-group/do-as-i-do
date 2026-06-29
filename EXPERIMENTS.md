# Do as I Do — Experiment Log

Paper: "Do as I Do: Dexterous Manipulation Data from Everyday Human Videos" (arXiv:2606.19333)  
Code: https://github.com/malik-group/do-as-i-do (MIT)  
Machine: RTX 4070 Laptop (8 GB VRAM), CUDA 12.1, Ubuntu 22.04, 32-core CPU, 31 GB RAM

---

## Module Feasibility Diagnosis (repo-recon)

| Module | Status | Blocker |
|--------|--------|---------|
| `reconstruction/` | INFEASIBLE (local) | README explicitly states ≥32 GB VRAM. Foundation models: SAM3, HaWoR, SAM3D each require substantial VRAM individually; running all four envs together exceeds 8 GB. Also requires HuggingFace auth for private `facebook/sam-3d-objects` and `facebook/sam3` repos, MANO license download, and an X display for interactive SAM3 segmentation (click-based GUI). |
| `retargeting/` | RUNNABLE (partial) | Single conda env (Python 3.12). Stages 1–4 (dataset processing, mesh decomp, scene gen, IK) are CPU-only. Stage 5 (MuJoCo Warp physics optimization) runs on GPU but fits in 8 GB. Full 1545-step run estimated at ~25 minutes; tested up to 1000 steps (~9 min) without OOM. |
| `deployment/` | INFEASIBLE | Requires physical dual UR3e arms + Sharpa Wave 22-DoF hands. No local execution path. |

---

## Environment Setup

### Prerequisites confirmed
- conda/mamba available: yes (miniforge3)
- CUDA: 12.1 system CUDA, but retargeting installs its own pip-based CUDA 13 runtime (self-contained, no conflict)
- Docker: not used; conda env suffices

### Retargeting environment (used for all experiments)

```bash
conda create -y -n retargeting python=3.12
conda activate retargeting
cd retargeting/
pip install -e .
```

Installed versions (key packages):
- `torch==2.12.1+cu130` — PyTorch with bundled CUDA 13 runtime
- `mujoco==3.4.0`
- `warp-lang==1.10.1` — MuJoCo Warp physics backend
- `mujoco-warp==0.0.1` — pinned commit from google-deepmind/mujoco_warp

Verify:
```bash
python -c "
import torch, mujoco, warp as wp, viser, retargeting
print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('mujoco:', mujoco.__version__)
print('warp:', wp.__version__)
print('viser, retargeting: OK')
"
# Expected output:
# torch: 2.12.1+cu130 CUDA: True NVIDIA GeForce RTX 4070 Laptop GPU
# mujoco: 3.4.0
# warp: 1.10.1
# viser, retargeting: OK
```

### MuJoCo Warp kernel compilation (one-time, ~75 seconds)
The first run of Stage 5 (optimize_physics) JIT-compiles Warp CUDA kernels. 
These are cached under `~/.cache/warp/1.10.1/`. Subsequent runs load from cache in <0.2s.

---

## Experiment 1: Pre-computed Whisking Demo Replay

**Goal:** Verify the repo-shipped pre-computed trajectory plays back correctly.  
**Pre-condition:** Repo must include upstream commits from after the fork (see Bootstrap section).

The repo ships a pre-computed whisking demo at:
- `retargeting/outputs/sharpa/right/whisking/0/trajectory_mjwp.npz` (2.8 MB)
- `retargeting/outputs/sharpa/right/whisking/0/scene.xml` (153 KB)

```bash
cd retargeting/
conda activate retargeting
timeout 30 python replay_viser.py --port 8081
```

**Result:** SUCCESS
```
viser listening *:8081
Scene:      outputs/sharpa/right/whisking/0/scene.xml
Trajectory: outputs/sharpa/right/whisking/0/trajectory_mjwp.npz
Playing 1000 frames (skipped 600 warmup frames; --no-skip-warmup to include)
Viewer running at http://localhost:8081  (Ctrl+C to exit)
```

The Sharpa Wave 22-DoF hand + whisk object scene loads and plays 1000 post-warmup frames.

---

## Experiment 2: Full Retargeting Pipeline (stages 1–5) on Whisking Demo

**Goal:** Run the complete pipeline end-to-end from pre-computed reconstruction data.

### Required input files
The reconstruction pipeline normally generates these files. They are **not committed to the repo** and must be created manually when using the pre-computed whisking reconstruction data:

**`reconstruction/whisking/config.json`** (written by `reconstruction/run_pipeline.sh`):
```json
{
    "frame_number": 125,
    "object_names": ["whisk"],
    "anchor_hand": "right"
}
```

**`reconstruction/whisking/gravity.json`** (written by `reconstruction/scripts/predict_video_gravity.py` via GeoCalib):
```json
{
    "roll_deg": 0.0,
    "pitch_deg": 0.0,
    "vec3d": [0.0, -1.0, 0.0],
    "n_frames": 138,
    "n_inliers": 130,
    "per_frame": []
}
```
Format: `vec3d` is world-up direction in camera frame (OpenCV convention: y-axis points down, so world-up = [0, -1, 0] for a level camera).

### Command
```bash
cd retargeting/
conda activate retargeting
python launch.py \
    --task whisking \
    --raw-dir ../reconstruction/whisking \
    --no-show-viewer \
    --no-wait-on-finish \
    --max-sim-steps 1000
```

### Stage-by-stage results

| Stage | Description | Time | Output |
|-------|-------------|------|--------|
| 1 | Dataset processing (gravity-align, spike clean, save trajectory_keypoints.npz) | ~0.3s | `outputs/mano/right/whisking/0/trajectory_keypoints.npz` |
| 2 | CoACD mesh decomposition (32 hulls, thicken 2mm, dilate 2mm) | ~24s | `outputs/assets/objects/whisking/convex/` |
| 3 | MJCF scene generation (602 contact pairs, UR3 arm cylinder) | ~0.5s | `outputs/sharpa/right/whisking/0/scene_ik.xml` |
| 4 | Inverse kinematics (mink, 137 frames) | ~0.2s | `outputs/sharpa/right/whisking/0/trajectory_kinematic.npz` |
| 4.5 | Pedestal resolution (object already in-hand, no pedestal needed) | ~1.0s | `outputs/sharpa/right/whisking/0/scene.xml` |
| 5 | Physics optimization (MuJoCo Warp, GPU) | ~9 min | `outputs/sharpa/right/whisking/0/trajectory_mjwp.npz` |

**Stage 5 details:**
- 1024 parallel environments × 32 MPPI iterations × 10 control steps (1000 physics steps total)
- First 600 physics steps = warmup (hand alignment with weld constraints, zero gravity)
- Remaining 400 steps = active physics optimization with perturb force
- Warp CUDA kernels: loaded from cache (~0.2s, first run takes 75s to compile)
- GPU VRAM: fits within 8 GB (measured idle VRAM before launch: 176 MiB)
- No OOM errors

**Result:** SUCCESS — trajectory saved, 400 playable frames

```bash
python replay_viser.py --port 8083
# Playing 400 frames (skipped 600 warmup frames; ...)
# Viewer running at http://localhost:8083
```

### Key constraint: max_sim_steps > warmup_steps
With `warmup_steps=600`, you need `max_sim_steps > 600` to get any playable post-warmup frames.
The minimum useful value is ~700 (600 warmup + 100 for 1 control step of actual optimization).
The original authors used `max_sim_steps=1545` for a 1000-frame trajectory.

---

## Reconstruction Module — Detailed Blockers (infeasible locally)

```
HARD BLOCKER: reconstruction/README.md explicitly states "An NVIDIA GPU with ≥ 32 GB VRAM"
Local machine has 8 GB VRAM → cannot run SAM3, SAM3D, HaWoR, or TAPIR.
```

Additional blockers even if VRAM were sufficient:
1. **HuggingFace private repos**: `facebook/sam-3d-objects` and `facebook/sam3` require HF auth token with model access approval.
2. **MANO license**: HaWoR requires a MANO hand model download from https://mano.is.tue.mpg.de (requires registration).
3. **X display required**: SAM3 segmentation uses a click-based GUI (`DISPLAY=:1`); headless requires VNC/Xvfb.
4. **4 separate conda environments**: sam3, sam3d, hawor, tapnet — each with different PyTorch/CUDA pins.

The pre-computed whisking reconstruction data already in the repo (video, hand meshes, object mesh, layout JSON) can be used as input to `retargeting/` without re-running reconstruction.

---

## Deployment Module — Infeasible

```
INFEASIBLE: requires physical hardware
- Dual UR3e robot arms
- Sharpa Wave 22-DoF dexterous hands
No simulation-only path exists for deployment/.
```

---

## Output File Inventory

After running Experiments 1 and 2:

```
retargeting/
  outputs/
    assets/
      objects/whisking/
        visual.obj          # scaled object mesh (0.1809× original)
        visual_texture.png
        convex/             # 32 CoACD convex hulls (0.obj ... 31.obj)
      robots/sharpa/meshes/ # STL robot meshes (used by scene.xml)
    mano/right/whisking/0/
      trajectory_keypoints.npz  # stage 1 output: cleaned MANO keypoints
    sharpa/right/whisking/0/
      scene_ik.xml          # pre-pedestal scene for IK
      scene.xml             # final scene (robot + object + UR3 wrist)
      scene_eq.xml          # scene with equality constraints
      trajectory_kinematic.npz  # stage 4 output: IK joint angles
      trajectory_ikrollout.npz  # stage 4 output: IK rollout
      trajectory_mjwp.npz   # stage 5 output: physics-optimized trajectory
      config.yaml           # resolved run configuration
```

---

## Replay Instructions (standalone)

```bash
cd do-as-i-do/retargeting/
conda activate retargeting

# Play pre-computed (pre-shipped) 1000-frame whisking demo:
python replay_viser.py --port 8081

# Play our freshly generated 400-frame trajectory:
python replay_viser.py --port 8082  # already points to outputs/sharpa/right/whisking/0/

# With warmup frames included:
python replay_viser.py --no-skip-warmup --port 8083
```

Open the printed URL in a browser. Use the Frame slider and Play button.
