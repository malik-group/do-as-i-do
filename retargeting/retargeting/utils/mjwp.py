"""Simulator for sampling with MuJoCo Warp (mjwarp)."""

from __future__ import annotations

from dataclasses import dataclass, field

import loguru
import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from retargeting.config import Config
from retargeting.utils.in_hand import (
    body_has_mesh_geom,
    compute_in_hand_mask,
    compute_near_floor_mask,
    compute_near_pedestal_mask,
    erode_mask,
    extract_body_mesh_verts,
)
from retargeting.utils.interp import align_to_sim_dt
from retargeting.utils.io import get_processed_data_dir
from retargeting.utils.math import quat_sub

# Initialize Warp once per process. warp>=1.13 no longer requires an explicit
# wp.init() (it initializes lazily) and can raise or be absent entirely, so
# tolerate every failure mode rather than only "already initialized".
try:
    wp.init()
except Exception:
    pass


@dataclass
class MJWPEnv:
    model_cpu: mujoco.MjModel
    data_cpu: mujoco.MjData
    model_wp: mjwarp.Model
    data_wp: mjwarp.Data
    data_wp_prev: mjwarp.Data
    # None when graph capture is unavailable (HIP/ROCm) — step_env runs eager.
    graph: "wp.ScopedCapture.Graph | None"
    # Device alias used for Warp allocations/launches (e.g., "cuda:1" or "cpu")
    device: str
    num_worlds: int

    # Per-object perturbation state, keyed by body id. Tensors are (N,1) bool /
    # (N,3) float and are populated lazily the first time apply_perturbation
    # runs for a given object.
    _perturb_active: dict[int, torch.Tensor] = field(default_factory=dict)
    _perturb_force: dict[int, torch.Tensor] = field(default_factory=dict)
    _perturb_torque: dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-object per-timestep gate for random perturbation. Keyed by object body
    # id; value is a (T,) bool tensor on `device` aligned with the (padded)
    # reference trajectory. A frame is True iff the reference holds the object
    # lifted off every rest surface (`_inhand_gate_mask & ~(pedestal | floor)`)
    # AND that combined signal survives `erode_mask` by perturb_gate_lag_steps —
    # i.e. the grasp has been stably airborne for the lag on both sides of the
    # frame. Consumed directly by apply_perturbation. Absent key => no gating.
    _perturb_gate_mask: dict[int, torch.Tensor] = field(default_factory=dict)
    # Per-object per-timestep raw in-hand gate (un-lagged). Keyed by object body
    # id; value is a (T,) bool tensor on `device`, True iff the reference hand is
    # within the in-hand distance of the object at that step. This is the signal
    # the rest/floor-mismatch penalties read as `in_hand_ref` — they need the
    # true per-frame state with no lag, so the branch flips the instant the hand
    # grabs or releases. The perturbation gate is derived from this.
    _inhand_gate_mask: dict[int, torch.Tensor] = field(default_factory=dict)
    # Per-object per-timestep gate for reference object-to-pedestal proximity.
    # Keyed by object body id; value is a (T,) bool tensor on `device`, True iff
    # the reference object's lowest vertex is within
    # `object_pedestal_distance_thresh` above a pedestal's top face at that step
    # (over its footprint). Used by the pedestal-mismatch penalty to restrict
    # each branch to frames where the reference is unambiguously off / on the
    # pedestal.
    _obj_pedestal_gate_mask: dict[int, torch.Tensor] = field(default_factory=dict)
    # Per-object per-timestep gate for reference object-to-floor proximity.
    # Keyed by object body id; value is a (T,) bool tensor on `device`, True iff
    # the reference object's lowest vertex is within `object_floor_distance_thresh`
    # of the floor plane at that step. Built only when the scene has an
    # object-collidable floor. Floor analog of `_obj_pedestal_gate_mask`.
    _obj_floor_gate_mask: dict[int, torch.Tensor] = field(default_factory=dict)
    # Map from side ("right"/"left") to that side's object body id. Populated
    # alongside _inhand_gate_mask. Used by the pedestal-mismatch reward to
    # look up the in-hand gate value for each side without rescanning names.
    _side_object_body_id: dict[str, int] = field(default_factory=dict)


def _compile_step(
    model_wp: mjwarp.Model, data_wp: mjwarp.Data
) -> "wp.ScopedCapture.Graph | None":
    """Warm up and (if supported) capture a CUDA graph running one mjwarp.step.

    Returns the captured graph on CUDA, or ``None`` when graph capture is
    unavailable (HIP/ROCm GPUs do not implement CUDA-graph capture). A ``None``
    graph signals ``step_env`` to fall back to an eager ``mjwarp.step`` — same
    physics, no graph replay. The two warmup steps always run so kernel
    modules are JIT-loaded before the first real step either way.
    """

    def _step_once():
        mjwarp.step(model_wp, data_wp)

    # Warmup required — CUDA forbids module loading inside a capture.
    _step_once()
    _step_once()
    wp.synchronize()
    try:
        with wp.ScopedCapture() as capture:
            _step_once()
        wp.synchronize()
        return capture.graph
    except Exception as exc:
        # HIP/ROCm (and any backend without graph capture) lands here; run eager.
        loguru.logger.warning(
            "CUDA-graph capture unavailable ({}); falling back to eager "
            "mjwarp.step (expected on HIP/ROCm).", type(exc).__name__,
        )
        return None


# --
# Key functions
# --


def setup_mj_model(config: Config) -> mujoco.MjModel:
    model_cpu = mujoco.MjModel.from_xml_path(config.model_path)
    model_cpu.opt.timestep = float(config.sim_dt)
    if config.embodiment_type in ["left", "right", "bimanual"]:
        # setup for hand
        model_cpu.opt.iterations = 100
        model_cpu.opt.ls_iterations = 200
        model_cpu.opt.o_solref = [0.02, 1.0]
        model_cpu.opt.o_solimp = [
            0.9,
            0.95,
            0.001,
            0.5,
            2,
        ]
        model_cpu.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model_cpu


def _object_mesh_body_id(model, side: str) -> int:
    """Body id of the object mesh the given hand interacts with.

    Returns ``{side}_object`` when it carries a collision mesh. For a bimanual
    shared object (do_as_i_do emits a meshed ``right_object`` plus a meshless
    ``left_object`` placeholder), the meshless side falls back to the other
    side's meshed body, so both hands resolve their hand-object distance
    checks against the same real geometry. Returns ``-1`` if neither side has
    a meshed object body.
    """
    own = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    if body_has_mesh_geom(model, own):
        return own
    other = "left" if side == "right" else "right"
    other_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{other}_object")
    if body_has_mesh_geom(model, other_id):
        return other_id
    return own


def _collidable_floor_geom_id(model) -> int:
    """Floor geom id if it collides with an object, else -1.

    The ``"floor"`` plane exists whenever object- or hand-floor collision is on,
    but only an object-collidable floor is a rest surface for the floor penalty.
    do_as_i_do has the plane (for hand collision) yet no object↔floor pair, so it must
    be excluded — detected here via the explicit collision pairs.
    """
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        return -1
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        if floor_id not in (g1, g2):
            continue
        other = g2 if g1 == floor_id else g1
        if "object" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""):
            return floor_id
    return -1


def _load_mano_keypoints_for_sim(
    config: Config, T_ref: int
) -> dict[str, dict[str, np.ndarray] | None]:
    """Load MANO hand-surface and object trajectory and align to ``qpos_ref``.

    The perturbation gate (and warmup analytical init) run along the
    IK-retargeted ``qpos_ref`` trajectory, which has been resampled to
    ``sim_dt`` and prepended with ``warmup_steps`` duplicated frames
    (see ``retargeting/utils/io.py:load_data``). We mirror that transformation on
    the MANO arrays so that mask index ``t`` corresponds to
    ``qpos_ref[t]`` for any side that has data.

    Hard-fails if the NPZ or the required ``mano_verts_{side}`` key is
    missing on a side that has ``qpos_obj_{side}`` — silent fall-back
    would hide a dataset-processor regression behind an all-False gate.

    Returns ``{side: {"mano_verts","qpos_obj"} or None}``. ``None`` means
    the npz didn't carry that side (e.g. unimanual data).
    """
    import os

    keypoint_dir = get_processed_data_dir(
        output_root_dir=os.path.abspath(config.output_root_dir),
        dataset_name=config.dataset_name,
        robot_type="mano",
        embodiment_type=config.embodiment_type,
        task=config.task,
        data_id=config.data_id,
    )
    npz_path = os.path.join(keypoint_dir, "trajectory_keypoints.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"MANO trajectory_keypoints.npz not found at {npz_path}. "
            f"The in-hand check requires this NPZ — re-run the dataset "
            f"processor for this task."
        )
    loaded = np.load(npz_path)

    def _align_and_pad(arr_np: np.ndarray) -> np.ndarray:
        """Resample to sim_dt + prepend warmup + pad horizon, matching load_data."""
        # align_to_sim_dt expects 2D (T, D); flatten trailing dims.
        orig_shape = arr_np.shape
        flat = arr_np.reshape(orig_shape[0], -1).astype(np.float32)
        t_arr = torch.from_numpy(flat)
        aligned = align_to_sim_dt(t_arr, config.ref_dt, config.sim_dt).numpy()
        out = aligned.reshape((aligned.shape[0],) + orig_shape[1:])
        if config.warmup_steps > 0:
            n = int(config.warmup_steps)
            out = np.concatenate([np.broadcast_to(out[:1], (n,) + out.shape[1:]), out], axis=0)
        for _ in range(config.horizon_steps + config.ctrl_steps):
            out = np.concatenate([out, out[-1:]], axis=0)
        # Trim/pad to exactly T_ref (load_data uses interp+concat which can be off
        # by one rounding step; clip rather than fail).
        if out.shape[0] > T_ref:
            out = out[:T_ref]
        elif out.shape[0] < T_ref:
            pad = T_ref - out.shape[0]
            out = np.concatenate(
                [out, np.broadcast_to(out[-1:], (pad,) + out.shape[1:])], axis=0,
            )
        return out

    result: dict[str, dict[str, np.ndarray] | None] = {}
    for side in ("right", "left"):
        if f"qpos_obj_{side}" not in loaded.files:
            result[side] = None
            continue
        mano_verts_key = f"mano_verts_{side}"
        if mano_verts_key not in loaded.files:
            raise RuntimeError(
                f"In-hand check requires '{mano_verts_key}' in {npz_path} "
                f"(world-frame MANO hand-surface vertices, shape "
                f"(T, V_hand, 3)). The dataset processor for this task did "
                f"not emit it."
            )
        hand_verts = loaded[mano_verts_key]
        # The dataset processor writes the mano_verts key for both sides but the unprocessed
        # side has shape (0, 0, 3). Treat empty as "side not processed";
        # downstream loops skip it via obj_body_id < 0.
        if hand_verts.shape[0] == 0:
            result[side] = None
            continue
        result[side] = {
            "mano_verts": _align_and_pad(hand_verts),
            "qpos_obj": _align_and_pad(loaded[f"qpos_obj_{side}"]),
        }
    return result


def setup_env(config: Config, ref_data: tuple[torch.Tensor, ...]) -> MJWPEnv:
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref_data
    qpos_init = qpos_ref[0]

    model_cpu = setup_mj_model(config)
    data_cpu = mujoco.MjData(model_cpu)
    arrs = (qpos_init, qvel_ref[0], ctrl_ref[0])
    data_cpu.qpos[:] = arrs[0].detach().cpu().numpy()
    data_cpu.qvel[:] = arrs[1].detach().cpu().numpy()
    data_cpu.ctrl[:] = arrs[2].detach().cpu().numpy()
    mujoco.mj_step(model_cpu, data_cpu)

    # Set Warp default device to match config to ensure kernels/modules load on it
    wp.set_device(str(config.device))
    dev = str(config.device)
    with wp.ScopedDevice(dev):
        default_model_wp = mjwarp.put_model(model_cpu)

        nworld_total = int(config.num_samples) * int(config.num_perturb_samples)
        default_data_wp = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=nworld_total,
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        data_wp_prev = mjwarp.put_data(
            model_cpu,
            data_cpu,
            nworld=nworld_total,
            nconmax=int(config.nconmax_per_env),
            njmax=int(config.njmax_per_env),
        )
        default_graph = _compile_step(default_model_wp, default_data_wp)

    env = MJWPEnv(
        model_cpu=model_cpu,
        data_cpu=data_cpu,
        model_wp=default_model_wp,
        data_wp=default_data_wp,
        data_wp_prev=data_wp_prev,
        graph=default_graph,
        device=dev,
        num_worlds=nworld_total,
    )

    # Precompute per-timestep reference gate masks. The raw in-hand and
    # object-surface proximity masks feed the pedestal/floor-mismatch penalties
    # un-lagged; the perturbation gate is then derived as the lag-eroded
    # "held and lifted off any rest surface" combination (built in the second
    # pass below). Constant perturb_force/perturb_torque is not gated.
    if (
        config.perturb_force_scale != 0
        or config.perturb_torque_scale != 0
        or config.pedestal_penalty_scale > 0.0
        or config.floor_penalty_scale > 0.0
    ):
        T_ref = int(qpos_ref.shape[0])
        mano_kp = _load_mano_keypoints_for_sim(config, T_ref)
        # Every pedestal geom (added to worldbody, so geom_pos is already
        # world-frame). Shared by all objects: the proximity gate uses "resting
        # on any pedestal", matching the `any_pedestal` contact check below.
        ped_ids = [
            i for i in range(model_cpu.ngeom)
            if (mujoco.mj_id2name(model_cpu, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            .startswith(("right_pedestal_", "left_pedestal_"))
        ]
        ped_pos = (
            np.asarray(model_cpu.geom_pos[ped_ids], dtype=np.float64)
            if ped_ids else np.zeros((0, 3), dtype=np.float64)
        )
        # Cylinder geom_size is [radius, half_height, 0]; the rest gate measures
        # height above the pedestal's top face, so both are needed.
        ped_size = (
            np.asarray(model_cpu.geom_size[ped_ids], dtype=np.float64)
            if ped_ids else np.zeros((0, 3), dtype=np.float64)
        )
        # Object-collidable floor (rest surface for the floor penalty), or -1.
        floor_id = _collidable_floor_geom_id(model_cpu)
        floor_z = (
            float(model_cpu.geom_pos[floor_id][2]) if floor_id >= 0 else 0.0
        )
        for side in ("right", "left"):
            if mujoco.mj_name2id(
                model_cpu, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object"
            ) < 0:
                continue
            # The hand-object distance check needs the real object mesh. For a
            # bimanual shared object only one side carries it (the other is a
            # meshless placeholder), so resolve to the meshed body — both
            # hands then gate against the same geometry.
            obj_body_id = _object_mesh_body_id(model_cpu, side)
            kp = mano_kp.get(side)
            if kp is None or obj_body_id < 0:
                loguru.logger.warning(
                    "Perturb gate ({}_object): MANO data missing for this "
                    "side — gate will be all-False.", side,
                )
                mask_np = np.zeros(T_ref, dtype=np.bool_)
                near_ped_np = np.zeros(T_ref, dtype=np.bool_)
                near_floor_np = np.zeros(T_ref, dtype=np.bool_)
            else:
                # apply_geom_xform=True puts verts in body-local frame —
                # what ``compute_in_hand_mask``'s per-frame R_obj/obj_pos
                # assumes. Without it, objects whose convex-decomp pieces
                # attach with non-identity per-geom transforms land in
                # wildly wrong world positions.
                obj_verts = extract_body_mesh_verts(
                    model_cpu, obj_body_id, apply_geom_xform=True,
                )
                mask_np = compute_in_hand_mask(
                    points_world=kp["mano_verts"],
                    qpos_obj=kp["qpos_obj"],
                    obj_verts=obj_verts,
                )
                # Reference object-to-pedestal proximity: object's lowest vertex
                # above the pedestal top face, the pedestal analog of the floor
                # check below (pedestals don't move, so pose is broadcast inside).
                near_ped_np = compute_near_pedestal_mask(
                    qpos_obj=kp["qpos_obj"],
                    obj_verts=obj_verts,
                    ped_pos=ped_pos,
                    ped_radius=ped_size[:, 0],
                    ped_half_h=ped_size[:, 1],
                    distance_thresh=config.object_pedestal_distance_thresh,
                )
                # Reference object-to-floor proximity: object's lowest vertex
                # within object_floor_distance_thresh of the floor plane.
                if floor_id >= 0:
                    near_floor_np = compute_near_floor_mask(
                        qpos_obj=kp["qpos_obj"],
                        obj_verts=obj_verts,
                        floor_z=floor_z,
                        distance_thresh=config.object_floor_distance_thresh,
                    )
                else:
                    near_floor_np = np.zeros(T_ref, dtype=np.bool_)
            frac = float(mask_np.mean()) if mask_np.size else 0.0
            loguru.logger.info(
                "In-hand gate ({}_object): in_hand on {}/{} frames ({:.1%}).",
                side, int(mask_np.sum()), mask_np.size, frac,
            )
            # When two sides share one physical object body (bimanual shared
            # object), OR their gates: the object is "held" if either hand
            # holds it.
            mask_t = torch.from_numpy(mask_np).to(dev)
            existing = env._inhand_gate_mask.get(obj_body_id)
            if existing is not None and existing.numel() == mask_t.numel():
                mask_t = mask_t | existing
            env._inhand_gate_mask[obj_body_id] = mask_t
            env._side_object_body_id[side] = obj_body_id
            # Object-to-pedestal proximity is a property of the (shared) object,
            # not the hand — OR across sides so the merged mask is identical
            # regardless of which side wrote it last.
            ped_t = torch.from_numpy(near_ped_np).to(dev)
            existing_ped = env._obj_pedestal_gate_mask.get(obj_body_id)
            if existing_ped is not None and existing_ped.numel() == ped_t.numel():
                ped_t = ped_t | existing_ped
            env._obj_pedestal_gate_mask[obj_body_id] = ped_t
            # Object-to-floor proximity, same per-object OR-merge as pedestal.
            floor_t = torch.from_numpy(near_floor_np).to(dev)
            existing_floor = env._obj_floor_gate_mask.get(obj_body_id)
            if existing_floor is not None and existing_floor.numel() == floor_t.numel():
                floor_t = floor_t | existing_floor
            env._obj_floor_gate_mask[obj_body_id] = floor_t

        # Build the perturbation gate from the merged per-object masks: hold the
        # object lifted off every rest surface, then erode both edges by the lag
        # so disturbances fire only well inside a stable airborne grasp. Done in
        # a second pass so in-hand and rest-surface proximity are both fully
        # merged across sides first. Warmup frames are forced off — the weld is
        # driving the object toward its init pose, not tracking the reference.
        lag = int(config.perturb_gate_lag_steps)
        warm = int(config.warmup_steps)
        for obj_body_id, inhand_t in env._inhand_gate_mask.items():
            near_rest = env._obj_pedestal_gate_mask[obj_body_id] | (
                env._obj_floor_gate_mask[obj_body_id]
            )
            held = (inhand_t & ~near_rest).cpu().numpy()
            perturb = erode_mask(held, lag)
            if warm > 0:
                perturb[:warm] = False
            loguru.logger.info(
                "Perturb gate (body {}): active on {}/{} frames ({:.1%}).",
                obj_body_id, int(perturb.sum()), perturb.size,
                float(perturb.mean()) if perturb.size else 0.0,
            )
            env._perturb_gate_mask[obj_body_id] = torch.from_numpy(perturb).to(dev)

    return env


def _build_qpos_group_slices(config: Config) -> dict[str, list[int]]:
    """Return named index lists into the nv-dim qpos_diff vector for per-group rewards.

    For bimanual, left and right hands are combined into single groups.
    """
    groups: dict[str, list[int]] = {}
    if config.embodiment_type == "bimanual":
        half_dof = int(config.nu // 2)
        groups["base_pos"] = list(range(0, 3)) + list(range(half_dof, half_dof + 3))
        groups["base_rot"] = list(range(3, 6)) + list(range(half_dof + 3, half_dof + 6))
        groups["joint"] = list(range(6, half_dof)) + list(range(half_dof + 6, config.nu))
        if config.nq_obj == 12:
            groups["obj_pos"] = list(range(config.nv - 12, config.nv - 9)) + list(range(config.nv - 6, config.nv - 3))
            groups["obj_rot"] = list(range(config.nv - 9, config.nv - 6)) + list(range(config.nv - 3, config.nv))
        else:
            groups["obj_pos"] = list(range(config.nv - 12, config.nv - 9)) + list(range(config.nv - 6, config.nv - 3))
            groups["obj_rot"] = list(range(config.nv - 9, config.nv - 6)) + list(range(config.nv - 3, config.nv))
    elif config.embodiment_type in ["right", "left"]:
        groups["base_pos"] = list(range(0, 3))
        groups["base_rot"] = list(range(3, 6))
        groups["joint"] = list(range(6, config.nu))
        if config.nq_obj == 6:
            groups["obj_pos"] = list(range(config.nv - 6, config.nv - 3))
            groups["obj_rot"] = list(range(config.nv - 3, config.nv))
        else:
            groups["obj_pos"] = list(range(config.nv - 6, config.nv - 3))
            groups["obj_rot"] = list(range(config.nv - 3, config.nv))
    return groups


def _weight_diff_qpos(config: Config) -> torch.Tensor:
    w = torch.ones(config.nv, device=config.device)
    if config.embodiment_type == "bimanual":
        half_dof = int(config.nu // 2)
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6:half_dof] = config.joint_rew_scale
        w[half_dof : half_dof + 3] = config.base_pos_rew_scale
        w[half_dof + 3 : half_dof + 6] = config.base_rot_rew_scale
        w[half_dof + 6 : config.nu] = config.joint_rew_scale
        # Object weights in nv-space: last 12 DOFs = 2 objects × (3 pos + 3 rot)
        w[-12:-9] = config.pos_rew_scale
        w[-9:-6] = config.rot_rew_scale
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    elif config.embodiment_type in ["right", "left"]:
        w[:3] = config.base_pos_rew_scale
        w[3:6] = config.base_rot_rew_scale
        w[6 : config.nu] = config.joint_rew_scale
        # Object weights in nv-space: last 6 DOFs = 1 object × (3 pos + 3 rot)
        w[-6:-3] = config.pos_rew_scale
        w[-3:] = config.rot_rew_scale
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return w


def _diff_qpos(
    config: Config, qpos_sim: torch.Tensor, qpos_ref: torch.Tensor
) -> torch.Tensor:
    batch_size = qpos_sim.shape[0]
    qpos_diff = torch.zeros((batch_size, config.nv), device=config.device)
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            qpos_diff[:, :-12] = qpos_sim[:, :-12] - qpos_ref[:, :-12]
            qpos_diff[:, -12:-9] = qpos_sim[:, -12:-9] - qpos_ref[:, -12:-9]
            qpos_diff[:, -9:-6] = qpos_sim[:, -9:-6] - qpos_ref[:, -9:-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        # joint (nq robot = nv robot, maps 1:1)
        qpos_diff[:, :-12] = qpos_sim[:, :-14] - qpos_ref[:, :-14]
        # position (nq[-14:-11] → nv[-12:-9], nq[-7:-4] → nv[-6:-3])
        qpos_diff[:, -12:-9] = qpos_sim[:, -14:-11] - qpos_ref[:, -14:-11]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        # rotation (nq[-11:-7] → nv[-9:-6], nq[-4:] → nv[-3:])
        qpos_diff[:, -9:-6] = quat_sub(qpos_sim[:, -11:-7], qpos_ref[:, -11:-7])
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            qpos_diff[:, :-6] = qpos_sim[:, :-6] - qpos_ref[:, :-6]
            qpos_diff[:, -6:-3] = qpos_sim[:, -6:-3] - qpos_ref[:, -6:-3]
            qpos_diff[:, -3:] = qpos_sim[:, -3:] - qpos_ref[:, -3:]
            return qpos_diff
        qpos_diff[:, :-6] = qpos_sim[:, :-7] - qpos_ref[:, :-7]
        qpos_diff[:, -6:-3] = qpos_sim[:, -7:-4] - qpos_ref[:, -7:-4]
        qpos_diff[:, -3:] = quat_sub(qpos_sim[:, -4:], qpos_ref[:, -4:])
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    return qpos_diff


def precompute_hand_object_geom_mask(config: Config, model_cpu: mujoco.MjModel):
    """Precompute boolean lookup tensors for hand/object geom classification.

    Stores _is_hand_geom and _is_object_geom on config for use by
    check_penetration during reward/termination computation. Also stores
    per-side object and pedestal masks used by the pedestal-mismatch penalty.
    """
    ngeom = model_cpu.ngeom
    is_hand = torch.zeros(ngeom, dtype=torch.bool, device=config.device)
    is_object = torch.zeros(ngeom, dtype=torch.bool, device=config.device)
    is_right_object = torch.zeros(ngeom, dtype=torch.bool, device=config.device)
    is_left_object = torch.zeros(ngeom, dtype=torch.bool, device=config.device)
    is_right_pedestal = torch.zeros(ngeom, dtype=torch.bool, device=config.device)
    is_left_pedestal = torch.zeros(ngeom, dtype=torch.bool, device=config.device)
    for i in range(ngeom):
        name = mujoco.mj_id2name(model_cpu, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "collision_hand_" in name:
            is_hand[i] = True
        if "object" in name:
            is_object[i] = True
            if name.startswith("right_object"):
                is_right_object[i] = True
            elif name.startswith("left_object"):
                is_left_object[i] = True
        # Stabilizer supports are welded to the object body and explicitly
        # contact the pedestal (the object's mesh geoms often don't reach the
        # pedestal directly). Treat them as part of the same-side object for
        # pedestal-contact detection. They use contype=0/conaffinity=0 so they
        # never appear in hand contacts — no false positives there.
        if name.startswith("right_support_"):
            is_right_object[i] = True
        elif name.startswith("left_support_"):
            is_left_object[i] = True
        if name.startswith("right_pedestal_"):
            is_right_pedestal[i] = True
        elif name.startswith("left_pedestal_"):
            is_left_pedestal[i] = True
    # Mark the floor as a rest surface only when objects can collide with it
    # (do_as_i_do has a hand-only floor that must stay out of this mask).
    is_floor = torch.zeros(ngeom, dtype=torch.bool, device=config.device)
    floor_id = _collidable_floor_geom_id(model_cpu)
    if floor_id >= 0:
        is_floor[floor_id] = True
    config._is_hand_geom = is_hand
    config._is_object_geom = is_object
    config._is_right_object_geom = is_right_object
    config._is_left_object_geom = is_left_object
    config._is_right_pedestal_geom = is_right_pedestal
    config._is_left_pedestal_geom = is_left_pedestal
    config._is_floor_geom = is_floor


def check_penetration(
    config: Config, env: MJWPEnv
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-contact hand-object penetration depths and their world IDs.

    The MJWarp contact buffer is flat: only the first nacon entries are active.
    Only hand-object contacts are included (filtered via precomputed geom masks).
    Both returned tensors are size-0 if there are no active contacts.
    """
    nacon = int(wp.to_torch(env.data_wp.nacon).item())
    if nacon == 0:
        return (
            torch.zeros(0, device=config.device),
            torch.zeros(0, dtype=torch.long, device=config.device),
        )

    # Read flat contact arrays; only first nacon entries are valid
    dist_flat = wp.to_torch(env.data_wp.contact.dist).reshape(-1)[:nacon]
    geom_flat = wp.to_torch(env.data_wp.contact.geom).reshape(-1, 2)[:nacon]
    worldid = wp.to_torch(env.data_wp.contact.worldid).reshape(-1)[:nacon]

    pen = torch.clamp(-dist_flat, min=0.0)
    pen = torch.nan_to_num(pen, nan=0.0)

    if hasattr(config, "_is_hand_geom"):
        ngeom = config._is_hand_geom.shape[0]
        g1 = geom_flat[:, 0].long().clamp(0, ngeom - 1)
        g2 = geom_flat[:, 1].long().clamp(0, ngeom - 1)
        ho_mask = (config._is_hand_geom[g1] & config._is_object_geom[g2]) | (
            config._is_object_geom[g1] & config._is_hand_geom[g2]
        )
        pen = pen * ho_mask

    return pen, worldid.long().clamp(0, env.num_worlds - 1)


def _pair_contact_worlds(
    env: MJWPEnv, a_mask: torch.Tensor, b_mask: torch.Tensor
) -> torch.Tensor:
    """Per-world bool tensor: any touching contact between geom-sets A and B.

    A world is True iff its contact buffer holds a contact between any geom in
    mask A and any geom in mask B with distance <= 0 (actually touching, not
    merely within margin). ``a_mask``/``b_mask`` are per-geom bool tensors of
    length ``ngeom``.
    """
    N = env.num_worlds
    device = a_mask.device
    out = torch.zeros(N, dtype=torch.bool, device=device)
    if not (bool(a_mask.any()) and bool(b_mask.any())):
        return out

    nacon = int(wp.to_torch(env.data_wp.nacon).item())
    if nacon == 0:
        return out

    dist_flat = wp.to_torch(env.data_wp.contact.dist).reshape(-1)[:nacon]
    geom_flat = wp.to_torch(env.data_wp.contact.geom).reshape(-1, 2)[:nacon]
    worldid = (
        wp.to_torch(env.data_wp.contact.worldid)
        .reshape(-1)[:nacon]
        .long()
        .clamp(0, N - 1)
    )
    touching = (dist_flat <= 0.0) & ~torch.isnan(dist_flat)

    ngeom = a_mask.shape[0]
    g1 = geom_flat[:, 0].long().clamp(0, ngeom - 1)
    g2 = geom_flat[:, 1].long().clamp(0, ngeom - 1)
    pair = (a_mask[g1] & b_mask[g2]) | (b_mask[g1] & a_mask[g2])
    hit = pair & touching
    if not bool(hit.any()):
        return out
    per_world = torch.zeros(N, device=device)
    per_world.scatter_reduce_(
        0, worldid, hit.float(), reduce="amax", include_self=True
    )
    return per_world > 0.5


def _euler_xyz_to_rotmat_batch(
    rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor,
) -> torch.Tensor:
    """Intrinsic XYZ Euler angles -> (N, 3, 3) rotation matrices."""
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    # R = Rx(rx) @ Ry(ry) @ Rz(rz)
    return torch.stack([
        cy * cz, -cy * sz, sy,
        sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy,
        -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy,
    ], dim=-1).reshape(-1, 3, 3)


# MANO palm geometry constants (from site definitions in right.xml / left.xml)
_RIGHT_PALM_OFFSET = torch.tensor([-0.0946604, -0.00147896, -0.00335754])
_LEFT_PALM_OFFSET = torch.tensor([0.0946604, -0.00147896, -0.00335754])
_PALM_NORMAL_LOCAL = torch.tensor([0.0, -1.0, 0.0])


def _get_palm_geometry(
    model_cpu: mujoco.MjModel, side: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read (palm_offset, palm_normal) from the compiled MuJoCo model, or None."""
    from retargeting.utils.mujoco_utils import get_palm_geometry_from_model

    result = get_palm_geometry_from_model(model_cpu, side)
    if result is None:
        return None
    offset, normal, _ = result
    return (
        torch.tensor(offset, dtype=torch.float32),
        torch.tensor(normal, dtype=torch.float32),
    )


def _build_hand_specs(
    config: Config,
    qpos_ref_0: torch.Tensor,
    env: MJWPEnv,
    device: torch.device,
) -> list[dict]:
    """Per-hand specs for analytical init pose placement.

    Returns one dict per hand with: side, pos_idx, rot_idx, palm_normal_local,
    obj_pos.  All tensors are placed on `device`.
    """
    nq = qpos_ref_0.shape[0]
    nq_obj = config.nq_obj
    robot_nq = nq - nq_obj

    right_geom = _get_palm_geometry(env.model_cpu, "right")
    left_geom = _get_palm_geometry(env.model_cpu, "left")
    r_normal = (right_geom[1] if right_geom else _PALM_NORMAL_LOCAL).to(device)
    l_normal = (left_geom[1] if left_geom else _PALM_NORMAL_LOCAL).to(device)

    specs: list[dict] = []
    if config.embodiment_type == "bimanual":
        half = robot_nq // 2
        if nq_obj == 14:
            right_obj_pos = qpos_ref_0[-14:-11].to(device)
            left_obj_pos = qpos_ref_0[-7:-4].to(device)
        else:
            right_obj_pos = qpos_ref_0[-12:-9].to(device)
            left_obj_pos = qpos_ref_0[-6:-3].to(device)
        specs.append({
            "side": "right",
            "pos_idx": [0, 1, 2],
            "rot_idx": [3, 4, 5],
            "palm_normal_local": r_normal,
            "obj_pos": right_obj_pos,
        })
        specs.append({
            "side": "left",
            "pos_idx": [half, half + 1, half + 2],
            "rot_idx": [half + 3, half + 4, half + 5],
            "palm_normal_local": l_normal,
            "obj_pos": left_obj_pos,
        })
    elif config.embodiment_type in ("right", "left"):
        if nq_obj == 7:
            obj_pos = qpos_ref_0[-7:-4].to(device)
        else:
            obj_pos = qpos_ref_0[-6:-3].to(device)
        normal = r_normal if config.embodiment_type == "right" else l_normal
        specs.append({
            "side": config.embodiment_type,
            "pos_idx": [0, 1, 2],
            "rot_idx": [3, 4, 5],
            "palm_normal_local": normal,
            "obj_pos": obj_pos,
        })
    return specs


def warmup_analytical_init(
    config: Config,
    env: MJWPEnv,
    qpos_ref_0: torch.Tensor,
    ctrl_ref_0: torch.Tensor,
    finger_qpos_indices: list[int],
) -> dict:
    """Translate each hand along -palm_normal until it clears the object.

    Solves per hand for the smallest offset t ≥ 0 along -n_world such that
    every robot-hand mesh vertex (FK'd at the closed-grasp pose
    ``qpos_ref[0]``) is at least ``warmup_min_clearance`` from every object
    mesh vertex. Hands are solved independently; no mjwarp forward /
    penetration check.

    Uses the robot's own mesh, not the MANO surface — MANO underestimates
    a chunky robot's reach in flat-palm grasps (e.g. closing a laptop lid),
    where palm and fingertips extend past it. Using the closed-grasp pose
    is conservative for the actual sim init, which has fingers zeroed and
    therefore sweeps further from the object.

    Per-pair math: with d = hand_pt − vert, α = n·d, perp² = ||d||² − α²,
    and clearance c, the pair constrains t only when ||d|| < c (currently
    inside clearance) and perp² < c² (back-off direction passes through
    violation). Then t_pair = max(0, α + √(c² − perp²)). The chosen t is
    the max over pairs.

    Writes the final pose into ``env.data_wp.qpos``, ``ctrl``, ``qvel``.
    """
    nw = env.num_worlds
    device = config.device

    base_qpos = qpos_ref_0.to(device).clone()
    base_ctrl = ctrl_ref_0.to(device).clone()
    base_qpos[finger_qpos_indices] = 0.0
    base_ctrl[finger_qpos_indices] = 0.0

    specs = _build_hand_specs(config, qpos_ref_0, env, device)
    clearance = float(config.warmup_min_clearance)

    # FK once at the closed-grasp pose; both hand-link and object verts are
    # read out from this single ``data`` below.
    fk_data = mujoco.MjData(env.model_cpu)
    fk_data.qpos[:] = qpos_ref_0.detach().cpu().numpy()
    mujoco.mj_kinematics(env.model_cpu, fk_data)

    hand_results: list[dict] = []
    for spec in specs:
        side = spec["side"]
        rot_idx = spec["rot_idx"]
        palm_normal_local = spec["palm_normal_local"]

        # World-frame palm normal from the reference wrist orientation.
        ex = base_qpos[rot_idx[0]:rot_idx[0] + 1]
        ey = base_qpos[rot_idx[1]:rot_idx[1] + 1]
        ez = base_qpos[rot_idx[2]:rot_idx[2] + 1]
        R = _euler_xyz_to_rotmat_batch(ex, ey, ez)[0]
        n_world_t = R @ palm_normal_local                                  # (3,) torch
        n_world = n_world_t.detach().cpu().numpy().astype(np.float64)      # (3,) numpy

        # Resolve to the meshed object body — for a bimanual shared object the
        # meshless placeholder side falls back to the real geometry.
        obj_body_id = _object_mesh_body_id(env.model_cpu, side)

        # Gather world-frame verts of every mesh-bearing body on this side
        # except the object. The base-DOF chain has no mesh so it's filtered
        # naturally by ``body_has_mesh_geom``.
        hand_parts = []
        for bid in range(env.model_cpu.nbody):
            name = mujoco.mj_id2name(env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if not name.startswith(f"{side}_") or bid == obj_body_id:
                continue
            if not body_has_mesh_geom(env.model_cpu, bid):
                continue
            hand_parts.append(extract_body_mesh_verts(
                env.model_cpu, bid, data=fk_data, apply_geom_xform=True,
            ))
        hand_pts = np.concatenate(hand_parts, axis=0) if hand_parts else np.zeros((0, 3))

        if hand_pts.shape[0] == 0 or obj_body_id < 0:
            loguru.logger.warning(
                "Warmup ({}): robot hand or object mesh unavailable; "
                "leaving hand at reference pose.", side,
            )
            t_chosen = 0.0
            closest_hand_idx = -1
            final_min_dist = float("nan")
            ok = False
        else:
            obj_verts_w = extract_body_mesh_verts(
                env.model_cpu, obj_body_id, data=fk_data, apply_geom_xform=True,
            )
            # Stride-subsample to keep the (Vh × Vo × 3) pairwise array bounded.
            if hand_pts.shape[0] > 4000:
                hand_pts = hand_pts[:: (hand_pts.shape[0] + 3999) // 4000]
            if obj_verts_w.shape[0] > 4000:
                obj_verts_w = obj_verts_w[:: (obj_verts_w.shape[0] + 3999) // 4000]

            # Closed-form solve for smallest t >= 0 such that
            #   ||(hand_pt − t·n) − vert|| >= clearance for every (hand_pt, vert).
            d = hand_pts[:, None, :] - obj_verts_w[None, :, :]   # (Vh, V, 3)
            n_dot_d = d @ n_world                                # (Vh, V)
            d_norm_sq = (d ** 2).sum(-1)                         # (Vh, V)
            perp_sq = d_norm_sq - n_dot_d ** 2                   # (Vh, V)
            delta = clearance * clearance - perp_sq              # (Vh, V)
            r_hi = n_dot_d + np.sqrt(np.maximum(delta, 0.0))     # (Vh, V)
            active = (delta > 0) & (d_norm_sq < clearance ** 2)
            t_pair = np.where(active, np.maximum(r_hi, 0.0), 0.0)
            flat_idx = int(t_pair.argmax())
            closest_hand_idx, _ = divmod(flat_idx, t_pair.shape[1])
            t_chosen = float(t_pair.flat[flat_idx])

            new_hand_pts = hand_pts - t_chosen * n_world
            final_min_dist = float(np.linalg.norm(
                new_hand_pts[:, None, :] - obj_verts_w[None, :, :], axis=-1
            ).min())
            ok = True

        hand_results.append({
            "side": side,
            "pos_idx": spec["pos_idx"],
            "n_world": n_world_t,
            "t_chosen": t_chosen,
            "ok": ok,
            "closest_hand_idx": int(closest_hand_idx),
            "final_min_dist": final_min_dist,
        })

    # Combine: apply each hand's chosen offset on top of base_qpos.
    final_qpos = base_qpos.clone()
    final_ctrl = base_ctrl.clone()
    for r in hand_results:
        idx = torch.tensor(r["pos_idx"], device=device, dtype=torch.long)
        offset = float(r["t_chosen"]) * r["n_world"]
        final_qpos[idx] = base_qpos[idx] - offset
        final_ctrl[idx] = final_qpos[idx]

    zero_qvel = torch.zeros(
        nw, env.model_cpu.nv, device=device, dtype=torch.float32
    )
    all_qpos = final_qpos.unsqueeze(0).expand(nw, -1).contiguous()
    all_ctrl = final_ctrl.unsqueeze(0).expand(nw, -1).contiguous()
    wp.copy(env.data_wp.qpos, wp.from_torch(all_qpos.float()))
    wp.copy(env.data_wp.ctrl, wp.from_torch(all_ctrl.float()))
    wp.copy(env.data_wp.qvel, wp.from_torch(zero_qvel))
    # Recompute cartesian body poses from the qpos just written. setup_env's
    # one CPU mj_step ran with the open hand at the grasp wrist, ejecting the
    # object; without this, data_wp.xpos keeps that ejected pose and
    # set_weld_target would anchor the weld there instead of at the reset qpos.
    with wp.ScopedDevice(env.device):
        mjwarp.kinematics(env.model_wp, env.data_wp)
    wp.synchronize()

    summary = ", ".join(
        "{} t={:.3f}m closest_hand_vert={} min_dist={:.4f}m ok={}".format(
            r["side"], r["t_chosen"], r["closest_hand_idx"],
            r["final_min_dist"], r["ok"],
        )
        for r in hand_results
    )
    loguru.logger.info("Warmup analytical init: {}", summary)

    return {
        "hands": hand_results,
        "final_qpos": final_qpos,
    }


def _get_object_z(config: Config, qpos_sim: torch.Tensor) -> torch.Tensor:
    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            return torch.stack([qpos_sim[:, -10], qpos_sim[:, -4]], dim=1)
        else:
            return torch.stack([qpos_sim[:, -12], qpos_sim[:, -5]], dim=1)
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            return qpos_sim[:, -4].unsqueeze(1)
        else:
            return qpos_sim[:, -5].unsqueeze(1)
    else:
        return torch.zeros(qpos_sim.shape[0], 1, device=qpos_sim.device)


def _ref_contact_state(
    env: MJWPEnv, obj_body_id: int, sim_step: int
) -> tuple[bool, bool, bool, bool]:
    """Per-frame reference contact booleans for an object body.

    Returns ``(has_gate, in_hand_ref, near_ped_ref, near_floor_ref)``:
      - ``has_gate``: the hand-object in-hand gate exists for this body.
      - ``in_hand_ref``: reference hand-object distance is under threshold.
      - ``near_ped_ref``: reference object-pedestal distance is under threshold.
      - ``near_floor_ref``: reference object-floor distance is under threshold.

    These are the *raw, un-lagged* reference signals — the rest/floor-mismatch
    penalty branches need the true per-frame state so they flip the instant the
    hand grabs or releases. The perturbation gate applies its own lag on top
    (see ``_perturb_gate_mask``); it is consumed directly, not via this helper.
    """
    gate = env._inhand_gate_mask.get(obj_body_id)
    has_gate = gate is not None and gate.numel() > 0
    in_hand_ref = (
        bool(gate[min(sim_step, gate.shape[0] - 1)]) if has_gate else False
    )
    ped_gate = env._obj_pedestal_gate_mask.get(obj_body_id)
    near_ped_ref = (
        bool(ped_gate[min(sim_step, ped_gate.shape[0] - 1)])
        if ped_gate is not None and ped_gate.numel() > 0
        else False
    )
    floor_gate = env._obj_floor_gate_mask.get(obj_body_id)
    near_floor_ref = (
        bool(floor_gate[min(sim_step, floor_gate.shape[0] - 1)])
        if floor_gate is not None and floor_gate.numel() > 0
        else False
    )
    return has_gate, in_hand_ref, near_ped_ref, near_floor_ref


def _rest_surface_penalty(
    reward: torch.Tensor,
    env: MJWPEnv,
    config: Config,
    sim_step: int,
    *,
    scale: float,
    surface_mask: torch.Tensor,
    surface: str,
) -> dict[str, torch.Tensor]:
    """Per-world rest/in-hand contact-mismatch penalty for one rest surface.

    Returns one ``(N,)`` penalty tensor per side, keyed by the deduped owning
    side (``"right"``/``"left"``); summing the values gives the total penalty.
    A bimanual shared object (do_as_i_do) dedups to a single entry under the first
    side that resolves it (``"right"``). Returns an empty dict when the penalty
    is disabled (``scale <= 0`` or this surface is absent from the scene); when
    enabled, every gated object gets an entry every step (zeros during warmup or
    on a frame where no branch is active) so plot series stay stable.

    Per object, penalize worlds where the actual contact state disagrees with
    what the reference prescribes. Two reference signals gate which branch (if
    any) is active on a given frame: the in-hand gate (reference hand close to
    reference object) and the surface-proximity gate (reference object close to
    this rest surface — a pedestal or the floor):
      - in-hand branch, active when in-hand AND not near this surface: the
        reference holds the object lifted, so penalize worlds with no
        hand-object contact.
      - rest branch, active when near this surface AND not in-hand: the
        reference sets the object down, so penalize worlds with no
        object-surface contact.
    Frames satisfying neither (pickup/placement transitions) are unpenalized.
    Zeroed during warmup: the weld holds the object near its init pose just
    above the surface, so contact distance is positive and the rest branch
    would fire spuriously.

    Iterated per physical object body, deduplicated via `seen`: a bimanual
    shared object (do_as_i_do) is pointed at by both hands, and iterating per side
    would double-count it.

    ``surface`` is ``"pedestal"`` or ``"floor"`` and selects which proximity
    gate from [[_ref_contact_state]] arms the rest branch.

    NOTE: a stricter contradiction-only variant is kept commented out below —
    it penalizes the *wrong* relationship being present rather than the *right*
    one being absent.
    """
    out: dict[str, torch.Tensor] = {}
    if not (scale > 0.0 and bool(surface_mask.any())):
        return out
    in_warmup = int(sim_step) < int(config.warmup_steps)
    seen: set[int] = set()
    for side in ("right", "left"):
        obj_body_id = env._side_object_body_id.get(side, -1)
        if obj_body_id < 0 or obj_body_id in seen:
            continue
        seen.add(obj_body_id)
        has_gate, in_hand_ref, near_ped_ref, near_floor_ref = _ref_contact_state(
            env, obj_body_id, sim_step
        )
        if not has_gate:
            continue
        pen = torch.zeros_like(reward)
        if not in_warmup:
            near_ref = near_floor_ref if surface == "floor" else near_ped_ref
            body_name = mujoco.mj_id2name(
                env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, obj_body_id
            ) or ""
            obj_mask = (
                config._is_right_object_geom if body_name.startswith("right")
                else config._is_left_object_geom
            )
            if in_hand_ref and not near_ref:
                mismatch = ~_pair_contact_worlds(env, config._is_hand_geom, obj_mask)
                pen = scale * mismatch.float()
            elif near_ref and not in_hand_ref:
                mismatch = ~_pair_contact_worlds(env, surface_mask, obj_mask)
                pen = scale * mismatch.float()
            # else: ambiguous frame (transition, or reference disagrees with
            # itself) — no branch active, pen stays zero.
        out[side] = pen
    return out


def get_reward(
    config: Config,
    env: MJWPEnv,
    ref: tuple[torch.Tensor, ...],
    sim_step: int = 0,
) -> torch.Tensor:
    """Non-terminal step reward for MJWP batched worlds.

    sim_step indexes the per-frame in-hand gate used by the pedestal-mismatch
    penalty (same gate as the random-perturbation gate).
    """
    qpos_ref, qvel_ref, ctrl_ref, contact_ref, contact_pos_ref = ref
    qpos_sim = wp.to_torch(env.data_wp.qpos)
    qvel_sim = wp.to_torch(env.data_wp.qvel)

    qpos_diff = _diff_qpos(
        config, qpos_sim, qpos_ref.unsqueeze(0).expand(qpos_sim.shape[0], -1)
    )
    if not hasattr(config, "_qpos_weight_cache"):
        config._qpos_weight_cache = _weight_diff_qpos(config)
    delta_qpos = qpos_diff * config._qpos_weight_cache
    qpos_dist = torch.norm(delta_qpos, p=2, dim=1)
    qvel_dist = torch.norm(qvel_sim - qvel_ref, p=2, dim=1)

    qpos_rew = -qpos_dist * 1.0
    qvel_rew = -config.vel_rew_scale * qvel_dist * 1.0

    if not hasattr(config, "_qpos_group_slices"):
        config._qpos_group_slices = _build_qpos_group_slices(config)
    qpos_group_rew = {}
    for name, idx in config._qpos_group_slices.items():
        qpos_group_rew[name] = -torch.norm(delta_qpos[:, idx], p=2, dim=1)

    if config.contact_rew_scale > 0.0 and len(config.contact_site_ids) > 0:
        site_xpos_torch = wp.to_torch(env.data_wp.site_xpos)
        contact_pos = site_xpos_torch[:, config.contact_site_ids]
        contact_dist = torch.norm(contact_pos - contact_pos_ref, p=2, dim=-1)
        contact_dist_masked = contact_dist * contact_ref.unsqueeze(0)
        contact_rew = -contact_dist_masked.sum(dim=1)
    else:
        contact_rew = 0.0

    reward = qpos_rew + qvel_rew + contact_rew

    # Margin so light surface contact (< margin) is not penalized, only deeper
    # clip-through penetration is.
    pen_penalty = torch.zeros_like(reward)
    if config.penetration_penalty_scale > 0.0:
        pen, wid = check_penetration(config, env)
        pen = torch.clamp(pen - config.penetration_margin, min=0.0)
        pen_sq = pen * pen
        pen_penalty = torch.zeros(env.num_worlds, device=config.device)
        pen_penalty.scatter_reduce_(0, wid, pen_sq, reduce="amax", include_self=True)
        pen_penalty = config.penetration_penalty_scale * pen_penalty
        reward = reward - pen_penalty

    drop_penalty = torch.zeros_like(reward)
    if config.drop_penalty_scale > 0.0:
        obj_z = _get_object_z(config, qpos_sim)
        drop = torch.clamp(config.drop_z_thresh - obj_z, min=0.0)
        drop_penalty = config.drop_penalty_scale * (drop * drop).sum(dim=-1)
        reward = reward - drop_penalty

    # Rest/in-hand contact-mismatch penalties: keep the simulated object's
    # contact state consistent with the reference, separately per rest surface
    # (pedestal vs floor). See ``_rest_surface_penalty`` for the branch logic.
    # The two are mutually exclusive per dataset (do_as_i_do → pedestal) but
    # computed independently with their own scale and gate.
    any_pedestal = config._is_right_pedestal_geom | config._is_left_pedestal_geom
    pedestal_pens = _rest_surface_penalty(
        reward, env, config, sim_step,
        scale=config.pedestal_penalty_scale,
        surface_mask=any_pedestal,
        surface="pedestal",
    )
    floor_pens = _rest_surface_penalty(
        reward, env, config, sim_step,
        scale=config.floor_penalty_scale,
        surface_mask=config._is_floor_geom,
        surface="floor",
    )
    for pens in (pedestal_pens, floor_pens):
        for pen in pens.values():
            reward = reward - pen

    info = {
        "qpos_dist": qpos_dist,
        "qvel_dist": qvel_dist,
        "qpos_rew": qpos_rew,
        "qvel_rew": qvel_rew,
        "pen_penalty": pen_penalty,
        "drop_penalty": drop_penalty,
        **{f"pedestal_{s}": p for s, p in pedestal_pens.items()},
        **{f"floor_{s}": p for s, p in floor_pens.items()},
        **qpos_group_rew,
    }
    return reward, info


def get_terminal_reward(
    config: Config,
    env: MJWPEnv,
    ref_slice: tuple[torch.Tensor, ...],
    sim_step: int = 0,
) -> torch.Tensor:
    """Terminal reward focusing on object tracking."""
    rew, info = get_reward(config, env, ref_slice, sim_step=sim_step)
    terminal_rew = config.terminal_rew_scale * rew
    return terminal_rew, info


def get_terminate(
    config: Config, env: MJWPEnv, ref_slice: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    qpos_sim = wp.to_torch(env.data_wp.qpos)

    if config.embodiment_type == "bimanual":
        if config.nq_obj == 12:
            obj_z = torch.stack([qpos_sim[:, -10], qpos_sim[:, -4]], dim=1)
        else:
            obj_z = torch.stack([qpos_sim[:, -12], qpos_sim[:, -5]], dim=1)
        # Ignore missing objects (ref pos near zero)
        qpos_ref = ref_slice[0]
        if config.nq_obj == 12:
            right_ref_pos = qpos_ref[-12:-9]
            left_ref_pos = qpos_ref[-6:-3]
        else:
            right_ref_pos = qpos_ref[-7:-4]
            left_ref_pos = qpos_ref[-14:-11]
        mask = []
        mask.append(not torch.all(right_ref_pos.abs() < 1e-4))
        mask.append(not torch.all(left_ref_pos.abs() < 1e-4))
        mask = torch.tensor(mask, device=config.device)
        obj_z = obj_z * mask.unsqueeze(0)
        terminate = (obj_z < config.terminate_z_threshold).any(dim=1)
    elif config.embodiment_type in ["right", "left"]:
        if config.nq_obj == 6:
            obj_z = qpos_sim[:, -4]
        else:
            obj_z = qpos_sim[:, -5]
        terminate = obj_z < config.terminate_z_threshold
    else:
        raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")

    terminate_z = terminate

    pen, wid = check_penetration(config, env)
    max_pen = torch.zeros(env.num_worlds, device=config.device)
    max_pen.scatter_reduce_(0, wid, pen, reduce="amax")
    terminate_pen = max_pen > config.terminate_penetration_threshold
    terminate = terminate_z | terminate_pen

    return terminate, {"terminate_z": terminate_z, "terminate_pen": terminate_pen}


def get_qpos(config: Config, env: MJWPEnv) -> torch.Tensor:
    return wp.to_torch(env.data_wp.qpos)


def set_qpos(config: Config, env: MJWPEnv, qpos: torch.Tensor):
    qpos = qpos.to(config.device)
    if qpos.dim() == 1:
        qpos = qpos.unsqueeze(0).repeat(env.num_worlds, 1)
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))
    zero_qvel = torch.zeros((env.num_worlds, env.model_cpu.nv), device=config.device)
    wp.copy(env.data_wp.qvel, wp.from_torch(zero_qvel))
    wp.copy(
        env.data_wp.time,
        wp.from_torch(
            torch.zeros(env.num_worlds, dtype=torch.float32, device=config.device)
        ),
    )


def get_qvel(config: Config, env: MJWPEnv) -> torch.Tensor:
    return wp.to_torch(env.data_wp.qvel)


def compute_contact_point_delta(
    contact_mask_step: torch.Tensor,
    contact_pos_ref_step: torch.Tensor,
    site_xpos: torch.Tensor,
    hand_contact_site_ids: list[int | None],
    contact_indices: list[int],
) -> torch.Tensor | None:
    """Mean contact position delta for a hand (current - reference)."""
    current_positions = []
    reference_positions = []
    for idx in contact_indices:
        if idx >= len(hand_contact_site_ids) or idx >= contact_pos_ref_step.shape[0]:
            continue
        sid = hand_contact_site_ids[idx]
        if sid is None or contact_mask_step[idx] <= 0.5:
            continue
        current_positions.append(site_xpos[sid])
        reference_positions.append(contact_pos_ref_step[idx])

    if not current_positions:
        return None

    current_mean = torch.stack(current_positions, dim=0).mean(dim=0)
    reference_mean = torch.stack(reference_positions, dim=0).mean(dim=0)
    return current_mean - reference_mean


def get_trace(config: Config, env: MJWPEnv) -> torch.Tensor:
    """Return per-world trace points used for visualization. Minimal default returns
    an empty trace set of shape (N, 0, 3) when not configured.
    """
    site_xpos = wp.to_torch(env.data_wp.site_xpos)  # (N, nsite, 3)
    return site_xpos[:, config.trace_site_ids, :]


def save_state(env: MJWPEnv):
    _copy_state(env.data_wp, env.data_wp_prev)
    return env


def load_state(env: MJWPEnv, state):
    _copy_state(env.data_wp_prev, env.data_wp)
    return env


def set_gravity(env: MJWPEnv, gravity: list[float]):
    g = wp.to_torch(env.model_wp.opt.gravity)  # (1, 3)
    g[0, 0] = gravity[0]
    g[0, 1] = gravity[1]
    g[0, 2] = gravity[2]


def set_weld_active(config: Config, env: MJWPEnv, active: bool):
    """Enable or disable warmup weld constraints.

    wp.to_torch shares GPU memory, so in-place writes propagate to the warp array.
    Must write to data_wp.eq_active (nworld, neq) which is what the step function
    reads at runtime, not model_wp.eq_active0 which is only the reset default.
    """
    if not config.warmup_weld_eq_ids:
        return
    # Update runtime state (actually used by mjwarp.step)
    eq_active = wp.to_torch(env.data_wp.eq_active)  # (nworld, neq)
    for eq_id in config.warmup_weld_eq_ids:
        eq_active[:, eq_id] = active
    # Also update model default so resets preserve the intended state
    eq_active0 = wp.to_torch(env.model_wp.eq_active0)  # (neq,)
    for eq_id in config.warmup_weld_eq_ids:
        eq_active0[eq_id] = active
    if active:
        set_weld_solver(config, env, [0.01, 1.0], [0.9995, 0.9995, 0.001, 0.5, 2.0])


def set_weld_solver(config: Config, env: MJWPEnv, solref: list[float], solimp: list[float]):
    if not config.warmup_weld_eq_ids:
        return
    eq_solref = wp.to_torch(env.model_wp.eq_solref)  # (1, neq, 2)
    eq_solimp = wp.to_torch(env.model_wp.eq_solimp)  # (1, neq, 5)
    for eq_id in config.warmup_weld_eq_ids:
        eq_solref[0, eq_id, :] = torch.tensor(solref, device=eq_solref.device)
        eq_solimp[0, eq_id, :] = torch.tensor(solimp, device=eq_solimp.device)


def set_weld_target(config: Config, env: MJWPEnv):
    """Set weld constraint targets to current object body poses (world 0).

    For a weld-to-world constraint, eq_data layout is:
        [0:3]  anchor (constant, set by compiler)
        [3:6]  relpose position
        [6:10] relpose quaternion (wxyz)
        [10]   torquescale

    Given body world pos/quat, the correct relpose is:
        relpose_pos  = R_body_inv @ (anchor - body_pos)
        relpose_quat = quat_conjugate(body_quat)
    """
    if not config.warmup_weld_eq_ids:
        return
    xpos = wp.to_torch(env.data_wp.xpos)    # (nworld, nbody, 3)
    xquat = wp.to_torch(env.data_wp.xquat)  # (nworld, nbody, 4) wxyz
    eq_data = wp.to_torch(env.model_wp.eq_data)  # (1, neq, 11)
    for eq_id in config.warmup_weld_eq_ids:
        body_id = env.model_cpu.eq_obj1id[eq_id]
        anchor = eq_data[0, eq_id, 0:3].clone()  # constant anchor from compiler
        pos = xpos[0, body_id]       # (3,)
        quat = xquat[0, body_id]     # (4,) wxyz
        # quaternion conjugate: [w, -x, -y, -z]
        quat_conj = quat.clone()
        quat_conj[1:] = -quat_conj[1:]
        # rotate (anchor - pos) by inverse body rotation
        v = anchor - pos
        # quaternion-vector rotation: v' = q * v * q_conj
        # using q = quat_conj (which IS the inverse rotation)
        w, x, y, z = quat_conj[0], quat_conj[1], quat_conj[2], quat_conj[3]
        t = 2.0 * torch.stack([
            y * v[2] - z * v[1],
            z * v[0] - x * v[2],
            x * v[1] - y * v[0],
        ])
        relpose_pos = v + w * t + torch.stack([
            y * t[2] - z * t[1],
            z * t[0] - x * t[2],
            x * t[1] - y * t[0],
        ])
        eq_data[0, eq_id, 3:6] = relpose_pos
        eq_data[0, eq_id, 6:10] = quat_conj


def apply_perturbation(config: Config, env: MJWPEnv, sim_step: int = 0):
    right_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "right_object"
    )
    left_obj_id = mujoco.mj_name2id(
        env.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "left_object"
    )
    xfrc_applied = wp.to_torch(env.data_wp.xfrc_applied)
    N = env.num_worlds
    device = xfrc_applied.device
    gravity_mag = abs(config.default_gravity[2]) if len(config.default_gravity) > 2 else 9.81

    for obj_id in [right_obj_id, left_obj_id]:
        if obj_id == -1:
            continue
        xfrc_applied[:, obj_id, :] = 0.0
        # Skip a meshless placeholder body (bimanual shared-object phantom):
        # it carries no real geometry, and its gate lives on the meshed
        # object body instead.
        if not body_has_mesh_geom(env.model_cpu, obj_id):
            continue

        # Per-timestep gate from reference trajectory. Same value across all
        # worlds (it's a property of the reference, not the sim sample). The
        # gate (``_perturb_gate_mask``) already encodes "held and off any rest
        # surface, eroded by the lag" — on only well inside a stable airborne
        # grasp. When gated off, skip both constant and random perturbation, and
        # clear any lingering "continuing" state so it can't restart across the
        # gap. Absent gate => ungated (perturb every step).
        gate = env._perturb_gate_mask.get(obj_id)
        if gate is not None and gate.numel() > 0 and not bool(
            gate[min(sim_step, gate.shape[0] - 1)]
        ):
            if obj_id in env._perturb_active:
                env._perturb_active[obj_id].zero_()
            continue

        xfrc_applied[:, obj_id, :3] += config.perturb_force
        xfrc_applied[:, obj_id, 3:] += config.perturb_torque

        if config.perturb_force_scale == 0 and config.perturb_torque_scale == 0:
            continue

        if obj_id not in env._perturb_active:
            env._perturb_active[obj_id] = torch.zeros(N, 1, dtype=torch.bool, device=device)
            env._perturb_force[obj_id] = torch.zeros(N, 3, device=device)
            env._perturb_torque[obj_id] = torch.zeros(N, 3, device=device)

        active = env._perturb_active[obj_id]
        # Common-random-numbers across control candidates: world layout is
        # n*K + k (control candidate n, perturbation seed k), so we draw K
        # independent values per step and tile across the N_ctrl candidates.
        # Variance reduction comes from averaging over the K seeds in the rollout.
        K = int(config.num_perturb_samples)
        N_ctrl = N // K
        # Independent draws so an already-active world can be hit with a fresh
        # perturbation (restart) that overrides the continue/off decision.
        rand_start = torch.rand(K, 1, device=device).repeat(N_ctrl, 1)
        rand_continue = torch.rand(K, 1, device=device).repeat(N_ctrl, 1)
        starting = rand_start < config.perturb_prob
        continuing = rand_continue < config.perturb_continue_prob
        continuing = active & ~starting & continuing

        # Sample new forces for worlds that are starting.
        # Direction is uniform on the sphere; magnitude is uniform in
        # [0, perturb_force_scale * mass * g] (and likewise for torque). Per-K
        # seed scalars are repeated across the N_ctrl candidates to keep the
        # common-random-numbers structure intact.
        if starting.any():
            obj_mass = env.model_cpu.body_mass[obj_id]
            force_dir = torch.randn(K, 3, device=device)
            force_dir = (force_dir / (force_dir.norm(dim=1, keepdim=True) + 1e-8)).repeat(N_ctrl, 1)
            force_mag_scale = torch.rand(K, 1, device=device).repeat(N_ctrl, 1)
            new_force = (
                force_dir
                * force_mag_scale
                * config.perturb_force_scale
                * obj_mass
                * gravity_mag
            )

            torque_dir = torch.randn(K, 3, device=device)
            torque_dir = (torque_dir / (torque_dir.norm(dim=1, keepdim=True) + 1e-8)).repeat(N_ctrl, 1)
            torque_mag_scale = torch.rand(K, 1, device=device).repeat(N_ctrl, 1)
            new_torque = (
                torque_dir * torque_mag_scale * config.perturb_torque_scale
            )
            env._perturb_force[obj_id] = torch.where(starting, new_force, env._perturb_force[obj_id])
            env._perturb_torque[obj_id] = torch.where(starting, new_torque, env._perturb_torque[obj_id])

        active = continuing | starting
        env._perturb_active[obj_id] = active

        # Apply the stored force for active worlds (already gated above by
        # the early-return path).
        xfrc_applied[:, obj_id, :3] += env._perturb_force[obj_id] * active
        xfrc_applied[:, obj_id, 3:] += env._perturb_torque[obj_id] * active

    wp.copy(env.data_wp.xfrc_applied, wp.from_torch(xfrc_applied))
    return env


def step_env(
    config: Config,
    env: MJWPEnv,
    ctrl_mujoco: torch.Tensor,
    perturb: bool = True,
    sim_step: int = 0,
):
    """Step all worlds with provided MuJoCo-format controls of shape (N, nu).

    ``sim_step`` indexes into the per-timestep perturbation gate mask; only
    consulted when ``perturb`` is True and stochastic perturbation is enabled.
    """
    if ctrl_mujoco.dim() == 1:
        ctrl_mujoco = ctrl_mujoco.unsqueeze(0).repeat(env.num_worlds, 1)
    # Ensure we operate on the correct CUDA context/device
    with wp.ScopedDevice(env.device):
        if perturb and (
            config.perturb_force != 0
            or config.perturb_torque != 0
            or config.perturb_force_scale != 0
            or config.perturb_torque_scale != 0
        ):
            env = apply_perturbation(config, env, sim_step=sim_step)
        wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl_mujoco.to(torch.float32)))
        if env.graph is not None:
            wp.capture_launch(env.graph)
        else:
            # HIP/ROCm eager fallback: replay the same single step directly.
            mjwarp.step(env.model_wp, env.data_wp)


def save_env_params(config: Config, env: MJWPEnv):
    # Only record which group is active; parameters are embedded in separate models.
    # currently we choose this solution since pair_margin has a huge virtual dimension,
    # convert it to torch would lead to OOM
    pair_margin = 0.0
    xy_offset = 0.0
    gravity = wp.to_torch(env.model_wp.opt.gravity)[0].clone()  # (3,)
    eq_active = wp.to_torch(env.data_wp.eq_active).clone()  # (nworld, neq)
    eq_solref = wp.to_torch(env.model_wp.eq_solref).clone()  # (1, neq, 2)
    eq_solimp = wp.to_torch(env.model_wp.eq_solimp).clone()  # (1, neq, 5)
    return {
        "pair_margin": pair_margin,
        "xy_offset": xy_offset,
        "gravity": gravity,
        "eq_active": eq_active,
        "eq_solref": eq_solref,
        "eq_solimp": eq_solimp,
    }


def load_env_params(config: Config, env: MJWPEnv, env_param: dict):
    """Load the simulation parameters (pair_margin, object xy_offset, gains)."""
    # DR margin is a per-pair *offset* from the XML baseline, always rewritten
    # from that baseline. Always-rewrite (rather than skip-on-zero) is what
    # restores the model after a DR rollout: pair_margin lives in model_wp, not
    # the saved state, so load_state never resets it — without this, the last
    # non-zero DR margin would leak into outer-loop execution. Offset 0 ==
    # exact baseline; non-zero offsets preserve relative per-pair XML margins.
    if "pair_margin" in env_param and config.npair > 0:
        pair_margin_single_np = (
            np.asarray(env.model_cpu.pair_margin, dtype=np.float32)
            + np.float32(env_param["pair_margin"])
        )

        pair_margin_override_wp = wp.from_numpy(
            pair_margin_single_np, dtype=wp.float32, device=config.device
        )

        # Stride trick: make Warp treat the single instance as num_worlds copies
        # without allocating any new memory.
        pair_margin_override_wp.strides = (0,) + pair_margin_override_wp.strides
        pair_margin_override_wp.shape = (
            env.num_worlds,
        ) + pair_margin_override_wp.shape
        pair_margin_override_wp.ndim += 1
        wp.copy(env.model_wp.pair_margin, pair_margin_override_wp)

    # update object position (NOTE: currently, xy_offset is only one scalar, which means we only update in the diagonal direction)
    if "xy_offset" in env_param:
        qpos_override_th = wp.to_torch(env.data_wp.qpos)
        if config.embodiment_type == "bimanual":
            qpos_override_th[:, -14:-12] = (
                qpos_override_th[:, -14:-12] + env_param["xy_offset"]
            )
            qpos_override_th[:, -12:-10] = (
                qpos_override_th[:, -12:-10] + env_param["xy_offset"]
            )
        elif config.embodiment_type in ["right", "left"]:
            qpos_override_th[:, -7:-5] = (
                qpos_override_th[:, -7:-5] + env_param["xy_offset"]
            )

        wp.copy(env.data_wp.qpos, wp.from_torch(qpos_override_th))

    if "kp" in env_param or "kd" in env_param:
        actuator_ids = config.object_actuator_ids
        if not actuator_ids:
            loguru.logger.warning(
                "Object actuator ids are empty; skipping kp/kd updates."
            )
        else:
            kp = env_param.get("kp")
            kd = env_param.get("kd")
            if kp is None or kd is None:
                loguru.logger.warning(
                    "Both kp and kd are required to update actuator gains; skipping."
                )
            else:
                kp_np = np.asarray(kp, dtype=np.float32)
                kd_np = np.asarray(kd, dtype=np.float32)
                if kp_np.ndim == 0:
                    kp_np = np.full((len(actuator_ids),), kp_np, dtype=np.float32)
                if kd_np.ndim == 0:
                    kd_np = np.full((len(actuator_ids),), kd_np, dtype=np.float32)
                if kp_np.shape[0] != len(actuator_ids) or kd_np.shape[0] != len(
                    actuator_ids
                ):
                    raise ValueError(
                        "kp/kd size mismatch for object actuators: "
                        f"kp={kp_np.shape}, kd={kd_np.shape}, "
                        f"expected={len(actuator_ids)}"
                    )

                # Update CPU model (used for viewer and as source of truth)
                env.model_cpu.actuator_gainprm[actuator_ids, 0] = kp_np
                env.model_cpu.actuator_biasprm[actuator_ids, 1] = -kd_np

                if hasattr(env.model_wp, "actuator_gainprm") and hasattr(
                    env.model_wp, "actuator_biasprm"
                ):
                    gain_full = np.array(
                        env.model_cpu.actuator_gainprm, dtype=np.float32
                    )
                    bias_full = np.array(
                        env.model_cpu.actuator_biasprm, dtype=np.float32
                    )
                    wp.copy(
                        env.model_wp.actuator_gainprm,
                        wp.from_numpy(
                            gain_full, dtype=wp.float32, device=config.device
                        ),
                    )
                    wp.copy(
                        env.model_wp.actuator_biasprm,
                        wp.from_numpy(
                            bias_full, dtype=wp.float32, device=config.device
                        ),
                    )
                else:
                    loguru.logger.warning(
                        "MJWarp model has no actuator_gainprm/biasprm; updated CPU model only."
                    )

    if "gravity" in env_param:
        g = wp.to_torch(env.model_wp.opt.gravity)  # (1, 3)
        g[0] = env_param["gravity"]
    if "eq_active" in env_param:
        eq_active = wp.to_torch(env.data_wp.eq_active)
        eq_active[:] = env_param["eq_active"]
    if "eq_solref" in env_param:
        eq_solref = wp.to_torch(env.model_wp.eq_solref)
        eq_solref[:] = env_param["eq_solref"]
    if "eq_solimp" in env_param:
        eq_solimp = wp.to_torch(env.model_wp.eq_solimp)
        eq_solimp[:] = env_param["eq_solimp"]

    return env


def _broadcast_state(data_wp, num_worlds: int):
    """Broadcast state from first world to all worlds."""
    qpos0 = wp.to_torch(data_wp.qpos)[:1]
    qvel0 = wp.to_torch(data_wp.qvel)[:1]
    time0 = wp.to_torch(data_wp.time)[:1]
    ctrl0 = wp.to_torch(data_wp.ctrl)[:1]

    # Handle time specially as it might be 1D
    if time0.dim() == 1:
        time_repeated = time0.repeat(num_worlds)
    else:
        time_repeated = time0.repeat(num_worlds, 1)

    wp.copy(data_wp.qpos, wp.from_torch(qpos0.repeat(num_worlds, 1)))
    wp.copy(data_wp.qvel, wp.from_torch(qvel0.repeat(num_worlds, 1)))
    wp.copy(data_wp.time, wp.from_torch(time_repeated))
    wp.copy(data_wp.ctrl, wp.from_torch(ctrl0.repeat(num_worlds, 1)))

    qacc0 = wp.to_torch(data_wp.qacc)[:1]
    wp.copy(data_wp.qacc, wp.from_torch(qacc0.repeat(num_worlds, 1)))

    act0 = wp.to_torch(data_wp.act)[:1]
    wp.copy(data_wp.act, wp.from_torch(act0.repeat(num_worlds, 1)))

    act_dot0 = wp.to_torch(data_wp.act_dot)[:1]
    wp.copy(data_wp.act_dot, wp.from_torch(act_dot0.repeat(num_worlds, 1)))

    qfrc_applied0 = wp.to_torch(data_wp.qfrc_applied)[:1]
    wp.copy(data_wp.qfrc_applied, wp.from_torch(qfrc_applied0.repeat(num_worlds, 1)))

    xfrc_applied0 = wp.to_torch(data_wp.xfrc_applied)[:1]
    wp.copy(data_wp.xfrc_applied, wp.from_torch(xfrc_applied0.repeat(num_worlds, 1, 1)))

    mocap_pos0 = wp.to_torch(data_wp.mocap_pos)[:1]
    wp.copy(data_wp.mocap_pos, wp.from_torch(mocap_pos0.repeat(num_worlds, 1, 1)))

    mocap_quat0 = wp.to_torch(data_wp.mocap_quat)[:1]
    wp.copy(data_wp.mocap_quat, wp.from_torch(mocap_quat0.repeat(num_worlds, 1, 1)))

    xpos0 = wp.to_torch(data_wp.xpos)[:1]
    wp.copy(data_wp.xpos, wp.from_torch(xpos0.repeat(num_worlds, 1, 1)))

    xquat0 = wp.to_torch(data_wp.xquat)[:1]
    wp.copy(data_wp.xquat, wp.from_torch(xquat0.repeat(num_worlds, 1, 1)))

    xmat0 = wp.to_torch(data_wp.xmat)[:1]
    wp.copy(data_wp.xmat, wp.from_torch(xmat0.repeat(num_worlds, 1, 1, 1)))

    geom_xpos0 = wp.to_torch(data_wp.geom_xpos)[:1]
    wp.copy(data_wp.geom_xpos, wp.from_torch(geom_xpos0.repeat(num_worlds, 1, 1)))

    geom_xmat0 = wp.to_torch(data_wp.geom_xmat)[:1]
    wp.copy(data_wp.geom_xmat, wp.from_torch(geom_xmat0.repeat(num_worlds, 1, 1, 1)))

    site_xpos0 = wp.to_torch(data_wp.site_xpos)[:1]
    wp.copy(data_wp.site_xpos, wp.from_torch(site_xpos0.repeat(num_worlds, 1, 1)))


def sync_env(config: Config, env: MJWPEnv, mj_data: mujoco.MjData):
    """Broadcast the state from first env to all envs."""
    _broadcast_state(env.data_wp, env.num_worlds)


def _copy_state(src: mjwarp.Data, dst: mjwarp.Data):
    """Copy core simulation state from src to dst.

    Only copies variables that define the simulation state. Derived quantities
    (spatial transforms, contacts, constraints, sensor/actuator data) are
    recomputed by mjwarp.step() and do not need to be saved/restored.
    """
    wp.copy(dst.qpos, src.qpos)
    wp.copy(dst.qvel, src.qvel)
    wp.copy(dst.qacc, src.qacc)
    wp.copy(dst.time, src.time)
    wp.copy(dst.ctrl, src.ctrl)
    wp.copy(dst.act, src.act)
    wp.copy(dst.act_dot, src.act_dot)
    wp.copy(dst.qacc_warmstart, src.qacc_warmstart)

    # Forces and applied forces (user-set, not recomputed by step)
    wp.copy(dst.qfrc_applied, src.qfrc_applied)
    wp.copy(dst.xfrc_applied, src.xfrc_applied)

    # Mocap data (user-set targets)
    wp.copy(dst.mocap_pos, src.mocap_pos)
    wp.copy(dst.mocap_quat, src.mocap_quat)

    return dst
