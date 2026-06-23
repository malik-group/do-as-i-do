#!/usr/bin/env python3
"""replay_viser.py — replay a retargeted trajectory in a viser viewer.

Loads a generated MuJoCo scene (`scene.xml`) and an optimized trajectory
(`trajectory_mjwp.npz`) from a retargeting run directory and plays it back in the
browser — reusing the pipeline's own `retargeting.utils.viser_viewer` geometry /
posing code. This is a *view-only* replay of an existing result: it does NOT
re-run the optimization (unlike `launch.py`, whose viewer is tied to Stage 5).

Run from the `retargeting/` directory in the `retargeting` conda env:

    conda activate retargeting

    # defaults to the whisking demo output on port 8081
    python replay_viser.py

    # or point at any run dir / explicit files / different port
    python replay_viser.py --run-dir outputs/sharpa/right/whisking/0 --port 8081
    python replay_viser.py --scene path/to/scene.xml --traj path/to/trajectory_mjwp.npz
    python replay_viser.py --no-skip-warmup        # also show the 600 warmup frames

Then open http://localhost:<port> and use the Frame slider / Play button.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from retargeting.utils import viser_viewer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-dir",
        default="outputs/sharpa/right/whisking/0",
        help="Run dir containing scene.xml + trajectory_mjwp.npz + config.yaml.",
    )
    p.add_argument("--scene", default=None, help="Override path to scene.xml.")
    p.add_argument("--traj", default=None, help="Override path to trajectory .npz.")
    p.add_argument("--port", type=int, default=8081, help="Viser server port.")
    p.add_argument("--fps", type=float, default=60.0, help="Initial playback FPS.")
    p.add_argument(
        "--skip-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the leading warmup frames (default: skip).",
    )
    return p.parse_args()


def _read_warmup_steps(config_yaml: Path) -> int:
    """Best-effort read of `warmup_steps` from the run's saved config.yaml."""
    if not config_yaml.exists():
        return 0
    try:
        from omegaconf import OmegaConf

        return int(OmegaConf.load(str(config_yaml)).get("warmup_steps", 0) or 0)
    except Exception:
        return 0


def load_qpos(traj_path: Path, model_nq: int, skip_warmup: bool, warmup_steps: int):
    """Load + flatten the optimized qpos to (T, nq), ordered by sim step.

    `trajectory_mjwp.npz` stores qpos chunked as (n_chunks, steps_per_chunk, nq);
    chunks are sequential executed segments, so we order by `sim_step` and flatten.
    """
    d = np.load(str(traj_path), allow_pickle=True)
    if "qpos" not in d.files:
        raise SystemExit(f"'qpos' not in {traj_path} (keys: {list(d.files)})")
    qpos = np.asarray(d["qpos"])
    if qpos.ndim == 3:
        if "sim_step" in d.files and len(d["sim_step"]) == qpos.shape[0]:
            qpos = qpos[np.argsort(np.asarray(d["sim_step"]).ravel())]
        qpos = qpos.reshape(-1, qpos.shape[-1])
    if qpos.shape[-1] != model_nq:
        raise SystemExit(
            f"qpos width {qpos.shape[-1]} != model.nq {model_nq} — "
            "scene.xml and trajectory .npz are from different runs?"
        )
    if skip_warmup and warmup_steps > 0:
        qpos = qpos[warmup_steps:]
    return np.ascontiguousarray(qpos, dtype=np.float64)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # flush progress even when piped to a file
    args = parse_args()
    run_dir = Path(args.run_dir)
    scene_path = Path(args.scene) if args.scene else run_dir / "scene.xml"
    traj_path = Path(args.traj) if args.traj else run_dir / "trajectory_mjwp.npz"
    for pth in (scene_path, traj_path):
        if not pth.exists():
            raise SystemExit(f"Not found: {pth}")

    warmup_steps = _read_warmup_steps(run_dir / "config.yaml")

    # Build the MuJoCo model + data (spec is needed by build_and_log_scene_from_spec).
    spec = mujoco.MjSpec.from_file(str(scene_path))
    model = spec.compile()
    data = mujoco.MjData(model)

    qpos = load_qpos(traj_path, model.nq, args.skip_warmup, warmup_steps)
    n_frames = len(qpos)
    if n_frames == 0:
        raise SystemExit("No frames to play.")

    # Spin up viser and upload the scene geometry (no reference ghost, no Stage-5 GUI).
    viser_viewer.init_viser(app_name="retargeting-replay", port=args.port)
    server = viser_viewer._get_server()
    body_ids = viser_viewer.build_and_log_scene_from_spec(
        spec, model, xml_path=scene_path, build_ref=False, build_gui=False
    )

    def show_frame(frame_idx: int) -> None:
        fi = max(0, min(n_frames - 1, int(frame_idx)))
        data.qpos[:] = qpos[fi]
        mujoco.mj_kinematics(model, data)  # populate body xpos/xquat for log_frame
        viser_viewer.log_frame(
            data, sim_time=0.0, viewer_body_entity_and_ids=body_ids, record=False
        )

    # --- GUI: Frame slider + Play/Pause + FPS ---
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    play_button = server.gui.add_button("Play")
    fps_slider = server.gui.add_slider(
        "FPS", min=1, max=120, step=1, initial_value=int(args.fps)
    )

    playing = {"on": False}
    suppress_cb = {"on": False}  # don't double-render during programmatic advance

    def set_button_label(text: str) -> None:
        for attr in ("name", "label"):  # viser version compatibility
            try:
                setattr(play_button, attr, text)
                return
            except Exception:
                pass

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress_cb["on"]:
            show_frame(frame_slider.value)

    @play_button.on_click
    def _(_) -> None:
        playing["on"] = not playing["on"]
        set_button_label("Pause" if playing["on"] else "Play")

    def playback_loop() -> None:
        while True:
            if playing["on"]:
                nxt = int(frame_slider.value) + 1
                if nxt >= n_frames:
                    nxt = 0
                suppress_cb["on"] = True
                frame_slider.value = nxt
                suppress_cb["on"] = False
                show_frame(nxt)
                time.sleep(1.0 / max(1.0, float(fps_slider.value)))
            else:
                time.sleep(0.05)

    threading.Thread(target=playback_loop, daemon=True).start()

    show_frame(0)
    skipped = warmup_steps if (args.skip_warmup and warmup_steps) else 0
    print(f"Scene:      {scene_path}")
    print(f"Trajectory: {traj_path}")
    print(
        f"Playing {n_frames} frames"
        + (f" (skipped {skipped} warmup frames; --no-skip-warmup to include)" if skipped else "")
    )
    print(f"Viewer running at http://localhost:{args.port}  (Ctrl+C to exit)")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
