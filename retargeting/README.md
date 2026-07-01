# Do as I Do · Retargeting

Retargets a reconstructed hand-object demo onto a **robot hand**. Given the output directory of
the [`reconstruction/`](../reconstruction/README.md) pipeline (MANO hand tracks + object mesh +
per-frame object poses), the pipeline cleans and gravity-aligns the trajectories, builds a MuJoCo
scene, solves inverse kinematics, and runs sampling-based MPC physics optimization (MuJoCo Warp)
to produce a physically-consistent robot hand + object trajectory.

## Layout

```
retargeting/
├── launch.py                # the sole entry point (5-stage pipeline)
├── pyproject.toml           # pip package (install with `pip install -e .`)
├── config/
│   ├── default.yaml         # optimizer/simulator defaults
│   └── override/do_as_i_do.yaml  # dataset-specific tuning
├── retargeting/             # the package
│   ├── config.py            # run configuration
│   ├── pipeline/            # the 5 pipeline stages, in run order
│   ├── utils/               # simulator, optimizer, viewer, shared helpers
│   └── assets/robots/       # robot models (sharpa, mano)
└── env/                     # conda env docs (see env/README.md)
```

## Requirements

- An NVIDIA GPU with CUDA (MuJoCo Warp runs the physics optimization on GPU). AMD ROCm GPUs
  (gfx942) are also supported — see [`env/rocm.md`](env/rocm.md).
- A browser for the viser viewer (the default visualization; serves a web UI).
- A reconstruction pipeline output directory as input (e.g. the whisking demo).

## Setup (one time)

```bash
cd retargeting
conda create -y -n retargeting python=3.12
conda activate retargeting
pip install -e .
```

See [`env/README.md`](env/README.md) for exact-pin details.

## Run

First run the reconstruction pipeline on a video (e.g. the whisking demo), then from the
`retargeting/` directory:

```bash
python launch.py --task whisking --raw-dir ../reconstruction/whisking
```

`--raw-dir` is the reconstruction output directory (the video's directory); `--task` is the video
name. Useful flags:

- `--robot-type sharpa` — target robot hand (default `sharpa`)
- `--no-show-viewer` — disable the viser viewer (headless runs)
- `--max-sim-steps N` — bound the optimization length (0 = full trajectory)
- `--no-wait-on-finish` — exit instead of keeping the viewer alive at the end

## Pipeline stages

| # | stage | module |
|---|---|---|
| 1 | dataset processing (clean + gravity-align → keypoint trajectory) | `retargeting/pipeline/process_dataset.py` |
| 2 | object convex decomposition (CoACD) | `retargeting/pipeline/decompose_mesh.py` |
| 3 | MJCF scene generation (robot + object + UR3 wrist cylinder) | `retargeting/pipeline/generate_scene.py` |
| 4 | inverse kinematics (mink) | `retargeting/pipeline/solve_ik.py` |
| 4.5 | pedestal resolution (`scene_ik.xml` → `scene.xml`) | `retargeting/pipeline/resolve_pedestal.py` |
| 5 | physics optimization (sampling-based MPC, MuJoCo Warp) | `retargeting/pipeline/optimize_physics.py` |

## Outputs

Written under `outputs/` (relative to the `retargeting/` directory):

- `outputs/mano/{hand}/{task}/0/trajectory_keypoints.npz` — cleaned reference keypoints (stage 1)
- `outputs/assets/objects/{object}/` — object meshes + convex decomposition (stages 1–2)
- `outputs/{robot}/{hand}/{task}/0/scene.xml` — generated MuJoCo scene (stages 3–4.5)
- `outputs/{robot}/{hand}/{task}/0/trajectory_kinematic.npz` — IK trajectory (stage 4)
- `outputs/{robot}/{hand}/{task}/0/trajectory_mjwp.npz` + `config.yaml` — optimized trajectory
  (with per-step tracking-error metrics stored in the `.npz`) and the resolved run config (stage 5)

## Visualize a retargeted trajectory

`replay_viser.py` plays back a finished run (`scene.xml` + `trajectory_mjwp.npz`) in an
interactive [viser](https://github.com/nerfstudio-project/viser) viewer **without re-running
the optimization** — it reuses the pipeline's own `retargeting.utils.viser_viewer`. From the
`retargeting/` directory in the `retargeting` env:

```bash
conda activate retargeting
python replay_viser.py                       # whisking demo → http://localhost:8081
```

Open the printed URL and use the **Frame** slider / **Play** button. Useful flags:

- `--run-dir DIR` — run directory to replay (default `outputs/sharpa/right/whisking/0`)
- `--no-skip-warmup` — include the leading warmup/settling frames (skipped by default)
- `--port N` — viser port (default `8081`)
- `--scene scene.xml --traj trajectory_mjwp.npz` — point at explicit files

The shipped `whisking` demo runs this out of the box (no pipeline run needed). It uses the run's
`scene.xml`, `trajectory_mjwp.npz`, and `config.yaml`, plus the meshes `scene.xml` references
under `outputs/assets/` (`objects/whisking/visual.obj` + `convex/`, and the referenced
`robots/sharpa/meshes/*.STL`).

## Credits & licenses

This codebase is built on [SPIDER](https://github.com/facebookresearch/spider). Physics optimization uses
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), IK uses
[mink](https://github.com/kevinzakka/mink), and visualization uses
[viser](https://github.com/nerfstudio-project/viser). We gratefully acknowledge the original authors.
