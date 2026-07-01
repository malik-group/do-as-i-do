# Retargeting environment

Unlike `reconstruction/` (four conda envs), retargeting runs in a **single conda env** installed
as a pip package:

```bash
conda create -y -n retargeting python=3.12
conda activate retargeting
pip install -e .          # from the retargeting/ directory
```

Notes:
- `mujoco-warp` is pinned to a known-good commit in `pyproject.toml`; it pulls in a compatible
  `mujoco` and `warp-lang`. An NVIDIA GPU with CUDA is required for the physics-optimization stage.
- **AMD / ROCm:** to run the physics stage on AMD GPUs (gfx942) instead, see [`rocm.md`](rocm.md)
  and the one-shot [`setup_rocm.sh`](setup_rocm.sh).
- The viser viewer (default) serves a web UI; open the printed URL in a browser.

## Exact pins

If dependency resolution drifts, `env/retargeting.yml` is a full `conda env export` of a
known-working env, to be treated mainly as a version reference for manual installation.
Regenerate it on a known-good machine with:

```bash
conda env export -n retargeting > env/retargeting.yml
```
