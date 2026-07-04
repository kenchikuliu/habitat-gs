#!/usr/bin/env python3
r"""Generate the Gaussian-avatar **dynamic-navigation** benchmark data for gs_scenes.

For each ``(scene, avatar)`` pair this produces, under ``<gs_scenes>/dynamic_nav/``::

    dynamic_nav.scene_dataset_config.json      dataset config (stage glob + navmesh + scene_instances)
    scenes/<scene>.scene_instance.json         avatar wiring (stage + navmesh + gaussian_avatars)
    trajectories/<scene>.driver.pkl            GAMMA-baked avatar walk (joint_mats + proxy_capsules)
    episodes/<split>/<split>.json.gz           content-shard index
    episodes/<split>/content/<scene>.json.gz   PointNav episodes, agent start near avatar start

The avatar trajectory (``driver.pkl``) is baked by ``tools_gs/generate_trajectory.py`` in the
``GAMMA`` conda env (learned motion synthesis along a navmesh path, biased toward the navmesh
centre so the human walks through open space, not along walls).  Everything else runs in the
``habitat-gs`` env (needs ``habitat_sim`` PathFinder only — no renderer).

Episodes are plain PointNav; the avatar is wired purely through the scene_instance.json.  The
agent start is sampled in an annulus around the avatar's start and is oriented to face the
avatar, and the point goal is biased toward the avatar's trajectory end so the agent traverses
the same region — giving both the *avoidance* and *tracking* rewards a meaningful interaction.

Usage (from the habitat-gs repo root)::

    # one scene, one avatar
    python scripts_gs/generate_dynamic_nav.py --scenes scene01 --avatars 2
    # several scenes; --avatars is matched per scene (or give a single id to broadcast)
    python scripts_gs/generate_dynamic_nav.py --scenes scene01 scene02 scene03 --avatars 2 1 6
    # skip trajectory baking (drivers already present)
    python scripts_gs/generate_dynamic_nav.py --scenes scene01 --avatars 2 --stages wiring episodes
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def find_conda() -> str:
    """Locate the ``conda`` executable robustly across environments."""
    cand = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if cand and os.path.exists(cand):
        return cand
    for base in (
        os.environ.get("CONDA_PREFIX", ""),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        "/mnt/data/home/ziyuan/miniforge3",
    ):
        p = os.path.join(base, "bin", "conda") if base else ""
        if p and os.path.exists(p):
            return p
    return "conda"  # last resort: rely on PATH at call time

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_EPISODES = 100
DEFAULT_VAL_EPISODES = 10

# Per-avatar SMPL-X gender used for motion synthesis + runtime skinning.  There is no
# per-avatar gender metadata shipped with the canonical_gs assets; a fixed, reasonable
# assignment keeps the baked skeleton consistent across regenerations.  Unknown avatar
# ids fall back to "female" (override with --genders).
AVATAR_GENDER: Dict[int, str] = {1: "female", 2: "female", 3: "male", 4: "male", 6: "female"}

# Uniform avatar scale (about the root joint; runtime scales splats + proxy capsules
# consistently). The raw canonical bodies stand only ~1.45-1.6 m; x1.15 puts them at a
# realistic 1.65-1.85 m against scene references (doors, lamps, benches).
AVATAR_SCALE: float = 1.15


def compute_offset_y(driver_pkl: str, navmesh_path: str, scale: float,
                     margin: float = 0.02) -> float:
    """Y-lift so the (scaled) avatar's feet walk on the navmesh ground surface.

    Uses the baked world-frame proxy capsules: foot sole = lowest capsule endpoint
    minus its radius, scaled about the root (the runtime scales the same way);
    ground = median navmesh height along the walk. offset = ground - sole + margin.
    """
    with open(driver_pkl, "rb") as f:
        d = pickle.load(f)
    t = np.asarray(d["transl"], np.float32)
    caps = np.asarray(d["proxy_capsules"], np.float32)
    y_min = np.minimum(caps[:, :, 1], caps[:, :, 4]) - caps[:, :, 6]
    # scale feet about the per-frame root (pelvis == transl)
    feet = t[:, 1] + (y_min.min(axis=1) - t[:, 1]) * scale
    feet_med = float(np.median(feet[::8]))
    pf = load_pathfinder(navmesh_path)
    grounds = []
    for k in range(0, t.shape[0], 8):
        p = np.asarray(pf.snap_point(t[k]), np.float32)
        if np.all(np.isfinite(p)):
            grounds.append(float(p[1]))
    ground = float(np.median(grounds))
    return round(ground - feet_med + margin, 3)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def repo_root() -> str:
    """habitat-gs repo root (this file lives in ``<root>/scripts_gs``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel_to(base_dir: str, target: str) -> str:
    """POSIX relative path from ``base_dir`` to ``target`` (for portable JSON refs)."""
    return os.path.relpath(target, base_dir).replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Navmesh utilities (habitat_sim PathFinder, no renderer)
# ---------------------------------------------------------------------------
def load_pathfinder(navmesh_path: str):
    import habitat_sim

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(navmesh_path)
    if not pf.is_loaded:
        raise RuntimeError(f"failed to load navmesh: {navmesh_path}")
    return pf


def geodesic(pf, start: np.ndarray, goal: np.ndarray) -> float:
    """Geodesic (shortest-navmesh-path) distance; ``inf`` when unreachable."""
    import habitat_sim

    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(start, np.float32)
    sp.requested_end = np.asarray(goal, np.float32)
    if not pf.find_path(sp):
        return float("inf")
    return float(sp.geodesic_distance)


def navmesh_center_ref(pf, num_samples: int = 4000, seed: int = 0) -> np.ndarray:
    """A high-clearance point near the centre of the largest navigable region.

    Used as ``--path-ref`` for the trajectory sampler so the avatar walks through open
    space rather than hugging walls.
    """
    rng = np.random.RandomState(seed)
    pts = np.asarray([pf.get_random_navigable_point() for _ in range(num_samples)], np.float32)
    # Keep the dominant island (largest island_radius) to avoid averaging across rooms.
    radii = np.asarray([pf.island_radius(p) for p in pts], np.float32)
    keep = pts[radii >= 0.75 * float(radii.max())] if radii.max() > 0 else pts
    centroid = pf.snap_point(keep.mean(axis=0))
    centroid = np.asarray(centroid, np.float32)
    if not np.all(np.isfinite(centroid)):
        centroid = np.asarray(pf.snap_point(pts.mean(axis=0)), np.float32)
    # Nudge toward the most open sampled point if the centroid sits in a tight spot.
    if pf.distance_to_closest_obstacle(centroid) < 1.0:
        clr = np.asarray([pf.distance_to_closest_obstacle(p) for p in keep], np.float32)
        centroid = np.asarray(pf.snap_point(keep[int(np.argmax(clr))]), np.float32)
    _ = rng  # reserved for future stochastic refs; keep signature stable
    return centroid


# ---------------------------------------------------------------------------
# Stage 1 — avatar trajectory (driver.pkl) via GAMMA
# ---------------------------------------------------------------------------
def bake_trajectory(
    *,
    navmesh_path: str,
    out_pkl: str,
    path_ref: np.ndarray,
    gender: str,
    gamma_root: str,
    gamma_env: str,
    conda_bin: str,
    tool: str,
    path_length: float,
    gamma_depth: int,
    fps: float,
    gpu_index: int,
    seed: int,
    log_path: str,
) -> None:
    data = os.path.join(gamma_root, "exp_GAMMAPrimitive", "data")
    cmd = [
        conda_bin, "run", "-n", gamma_env, "python", tool,
        "--navmesh", navmesh_path,
        "--output", out_pkl,
        "--path-length", f"{path_length:.3f}",
        "--path-ref", f"{path_ref[0]:.4f}", f"{path_ref[1]:.4f}", f"{path_ref[2]:.4f}",
        "--gamma-root", gamma_root,
        "--body-model-path", os.path.join(data, "VPoser"),
        "--marker-path", os.path.join(data, "Mosh"),
        "--smpl-model-path", os.path.join(data, "VPoser", "smplx"),
        "--smpl-type", "smplx",
        "--gender", gender,
        "--length", str(gamma_depth),
        "--fps", f"{fps:.1f}",
        "--include-proxy", "true",
        "--smooth-spikes",
        "--gpu-index", str(gpu_index),
        "--random-seed", str(seed),
    ]
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(log_path, "w") as lf:
        lf.write("COMMAND: " + " ".join(cmd) + "\n\n")
        lf.flush()
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0 or not os.path.exists(out_pkl):
        raise RuntimeError(f"trajectory baking failed (rc={rc}); see {log_path}")


def driver_start_xyz(driver_pkl: str) -> np.ndarray:
    with open(driver_pkl, "rb") as f:
        d = pickle.load(f)
    transl = np.asarray(d["transl"], np.float32)
    return transl[0].copy()


def driver_end_xyz(driver_pkl: str) -> np.ndarray:
    with open(driver_pkl, "rb") as f:
        d = pickle.load(f)
    transl = np.asarray(d["transl"], np.float32)
    return transl[-1].copy()


# ---------------------------------------------------------------------------
# Stage 2 — scene_instance.json (avatar wiring)
# ---------------------------------------------------------------------------
def write_stage_config(*, scene: str, gs_scenes_dir: str, stages_out_dir: str,
                       scene_split: str = "train") -> str:
    """Explicit GS stage template so the scene_instance can reference it by ``scene`` name."""
    os.makedirs(stages_out_dir, exist_ok=True)
    stage_path = os.path.join(stages_out_dir, f"{scene}.stage_config.json")
    gs_ply = os.path.join(gs_scenes_dir, scene_split, scene, f"{scene}.gs.ply")
    stage = {
        "render_asset": rel_to(stages_out_dir, gs_ply),
        "render_asset_type": "gaussian_splatting",
        "units_to_meters": 1.0,
        "orient_up": [0, 1, 0],
        "orient_front": [0, 0, -1],
        "frustum_culling": False,
        "light_setup": "default",
    }
    with open(stage_path, "w") as f:
        json.dump(stage, f, indent=2)
    return stage_path


def write_scene_instance(
    *, scene: str, avatar_id: int, gs_scenes_dir: str, scenes_out_dir: str,
    driver_pkl: str, offset_y: float, scale: float,
) -> str:
    inst_path = os.path.join(scenes_out_dir, f"{scene}.scene_instance.json")
    canonical = os.path.join(gs_scenes_dir, "avatars", f"avatar{avatar_id}", "canonical_gs.npz")
    smplx_dir = os.path.join(gs_scenes_dir, "avatars", "smplx")
    inst = {
        "stage_instance": {"template_name": scene},
        "navmesh_instance": scene,
        "gaussian_avatars": [
            {
                "canonical_gaussians": rel_to(scenes_out_dir, canonical),
                "driver": rel_to(scenes_out_dir, driver_pkl),
                "smpl_model_path": rel_to(scenes_out_dir, smplx_dir),
                "smpl_type": "smplx",
                "offset_y": offset_y,
                "scale": scale,
            }
        ],
    }
    os.makedirs(scenes_out_dir, exist_ok=True)
    with open(inst_path, "w") as f:
        json.dump(inst, f, indent=2)
    return inst_path


def write_dataset_config(*, dynamic_root: str, gs_scenes_dir: str, scenes: List[str],
                         scene_split: str = "train") -> str:
    """Scene dataset config for the dynamic-nav scenes.

    Merges the given scenes into any existing config (so per-scene invocations
    accumulate instead of overwriting). Paths are relative to the config file's
    directory (``dynamic_nav/``).
    """
    cfg_path = os.path.join(dynamic_root, "dynamic_nav.scene_dataset_config.json")
    gs_from_cfg = rel_to(dynamic_root, gs_scenes_dir)  # ".."
    navmeshes: Dict[str, str] = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                navmeshes = dict(json.load(f).get("navmesh_instances", {}))
        except (OSError, ValueError):
            pass
    for s in scenes:
        navmeshes[s] = f"{gs_from_cfg}/{scene_split}/{s}/{s}.navmesh"
    cfg = {
        "stages": {"paths": {".json": ["stages"]}},
        "scene_instances": {"paths": {".json": ["scenes"]}},
        "navmesh_instances": dict(sorted(navmeshes.items())),
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg_path


# ---------------------------------------------------------------------------
# Stage 3 — PointNav episodes coupled to the avatar
# ---------------------------------------------------------------------------
def _yaw_quat_facing(from_xz: np.ndarray, to_xz: np.ndarray) -> List[float]:
    """Habitat [x,y,z,w] quaternion whose -Z forward points from ``from`` to ``to``."""
    dx, dz = float(to_xz[0] - from_xz[0]), float(to_xz[1] - from_xz[1])
    if abs(dx) < 1e-8 and abs(dz) < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    theta = math.atan2(-dx, -dz)  # forward = (-sinθ, 0, -cosθ)
    return [0.0, math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0)]


def _sample_start_near(
    pf, avatar_xyz: np.ndarray, *, rmin: float, rmax: float, rng: np.random.RandomState,
    tries: int = 400,
) -> Optional[np.ndarray]:
    ax, ay, az = float(avatar_xyz[0]), float(avatar_xyz[1]), float(avatar_xyz[2])
    for _ in range(tries):
        r = rng.uniform(rmin, rmax)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        cand = np.asarray([ax + r * math.cos(ang), ay, az + r * math.sin(ang)], np.float32)
        snapped = np.asarray(pf.snap_point(cand), np.float32)
        if not np.all(np.isfinite(snapped)):
            continue
        if abs(float(snapped[1]) - ay) > 0.6:  # different floor
            continue
        planar = math.hypot(float(snapped[0]) - ax, float(snapped[2]) - az)
        if rmin - 0.25 <= planar <= rmax + 0.5:
            return snapped
    return None


def _sample_goal(
    pf, start: np.ndarray, avatar_end: np.ndarray, *, gmin: float, gmax: float,
    rng: np.random.RandomState, tries: int = 400,
) -> Optional[Tuple[np.ndarray, float]]:
    """Goal biased toward the avatar's trajectory end, with geodesic-distance gating."""
    ex, ey, ez = float(avatar_end[0]), float(avatar_end[1]), float(avatar_end[2])
    for i in range(tries):
        # First half: near the avatar end (so the agent traverses the avatar's route);
        # second half: anywhere navigable (fallback for reachability).
        if i < tries // 2:
            r = rng.uniform(0.0, 3.0)
            ang = rng.uniform(0.0, 2.0 * math.pi)
            cand = np.asarray([ex + r * math.cos(ang), ey, ez + r * math.sin(ang)], np.float32)
            goal = np.asarray(pf.snap_point(cand), np.float32)
        else:
            goal = np.asarray(pf.get_random_navigable_point(), np.float32)
        if not np.all(np.isfinite(goal)):
            continue
        geo = geodesic(pf, start, goal)
        if not math.isfinite(geo) or geo < gmin or geo > gmax:
            continue
        euc = float(np.linalg.norm((start - goal)[[0, 2]]))
        if euc > 0 and geo / euc < 1.05:  # reject near-straight-line trivial episodes
            continue
        return goal, float(geo)
    return None


def generate_episodes(
    *, scene: str, navmesh_path: str, driver_pkl: str, num_episodes: int, split: str,
    scene_id_rel: str, scene_dataset_config: str, out_dir: str, rng_seed: int,
    spawn_min: float, spawn_max: float, goal_min: float, goal_max: float,
    success_radius: float,
) -> str:
    pf = load_pathfinder(navmesh_path)
    avatar_start = driver_start_xyz(driver_pkl)
    avatar_end = driver_end_xyz(driver_pkl)
    rng = np.random.RandomState(rng_seed)

    episodes = []
    attempts = 0
    max_attempts = num_episodes * 60
    while len(episodes) < num_episodes and attempts < max_attempts:
        attempts += 1
        start = _sample_start_near(
            pf, avatar_start, rmin=spawn_min, rmax=spawn_max, rng=rng
        )
        if start is None:
            continue
        goal = _sample_goal(
            pf, start, avatar_end, gmin=goal_min, gmax=goal_max, rng=rng
        )
        if goal is None:
            continue
        goal_pos, geo = goal
        episodes.append(
            {
                "episode_id": str(len(episodes)),
                "scene_id": scene_id_rel,
                "scene_dataset_config": scene_dataset_config,
                "start_position": [float(x) for x in start],
                "start_rotation": _yaw_quat_facing(start[[0, 2]], avatar_start[[0, 2]]),
                "info": {"geodesic_distance": geo},
                "goals": [{"position": [float(x) for x in goal_pos], "radius": success_radius}],
                "start_room": None,
                "shortest_paths": None,
            }
        )

    if len(episodes) < num_episodes:
        print(
            f"  [WARN] {scene}/{split}: only {len(episodes)}/{num_episodes} episodes "
            f"after {attempts} attempts (navmesh may be tight around the avatar).",
            file=sys.stderr,
        )

    content_dir = os.path.join(out_dir, split, "content")
    os.makedirs(content_dir, exist_ok=True)
    shard = os.path.join(content_dir, f"{scene}.json.gz")
    with gzip.open(shard, "wt") as f:
        json.dump({"episodes": episodes}, f)
    return shard


def write_split_index(*, out_dir: str, split: str) -> str:
    """Top-level content-shard index (episodes live in ``content/<scene>.json.gz``)."""
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    idx = os.path.join(split_dir, f"{split}.json.gz")
    with gzip.open(idx, "wt") as f:
        json.dump({"episodes": []}, f)
    return idx


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="scene names, e.g. scene01 scene05")
    ap.add_argument("--avatars", nargs="+", type=int, default=[1],
                    help="avatar id per scene (or a single id broadcast to all scenes)")
    ap.add_argument("--genders", nargs="+", default=None,
                    choices=["male", "female"],
                    help="override the per-avatar gender (one per scene, or a single value)")
    ap.add_argument("--scene-split", default="train", choices=["train", "val"],
                    help="gs_scenes subfolder the scene assets live in (default: train)")
    ap.add_argument(
        "--stages", nargs="+", default=["trajectory", "wiring", "episodes"],
        choices=["trajectory", "wiring", "episodes"], help="which stages to run",
    )
    ap.add_argument("--gs-scenes", default=os.path.join(root, "data/scene_datasets/gs_scenes"))
    ap.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    ap.add_argument("--val-episodes", type=int, default=DEFAULT_VAL_EPISODES)
    ap.add_argument(
        "--splits", nargs="+", default=["train", "val"], choices=["train", "val"],
        help="which episode splits to (re)generate",
    )
    ap.add_argument("--offset-y", type=float, default=None,
                help="override the computed avatar Y-lift (default: computed per scene)")
    ap.add_argument("--scale", type=float, default=AVATAR_SCALE,
                help="avatar scale about the root joint (default: %(default)s)")
    ap.add_argument("--success-radius", type=float, default=0.2)
    ap.add_argument("--spawn-min", type=float, default=1.5, help="agent spawn annulus min (m) around avatar start")
    ap.add_argument("--spawn-max", type=float, default=4.0)
    ap.add_argument("--goal-min", type=float, default=3.0, help="geodesic goal distance min (m)")
    ap.add_argument("--goal-max", type=float, default=20.0)
    # trajectory / GAMMA
    ap.add_argument(
        "--gamma-root",
        default=next(
            (p for p in (os.environ.get("GAMMA_ROOT", ""),
                         os.path.expanduser("~/GAMMA-release"),
                         "/mnt/data/home/ziyuan/GAMMA-release")
             if p and os.path.isdir(p)),
            os.path.expanduser("~/GAMMA-release"),
        ),
    )
    ap.add_argument("--gamma-env", default="GAMMA")
    ap.add_argument("--conda-bin", default=find_conda())
    ap.add_argument("--traj-tool", default=os.path.join(root, "tools_gs/generate_trajectory.py"))
    ap.add_argument("--path-length", type=float, default=12.0)
    ap.add_argument("--gamma-depth", type=int, default=40)
    ap.add_argument("--fps", type=float, default=40.0)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes = list(args.scenes)
    avatars = list(args.avatars)
    if len(avatars) == 1:
        avatars = avatars * len(scenes)
    if len(avatars) != len(scenes):
        ap.error(f"--avatars must give 1 id or one per scene "
                 f"({len(scenes)} scenes, {len(avatars)} avatar ids)")
    genders = list(args.genders) if args.genders else None
    if genders and len(genders) == 1:
        genders = genders * len(scenes)
    if genders and len(genders) != len(scenes):
        ap.error("--genders must give 1 value or one per scene")

    gs = args.gs_scenes
    dynamic_root = os.path.join(gs, "dynamic_nav")
    scenes_out = os.path.join(dynamic_root, "scenes")
    stages_out = os.path.join(dynamic_root, "stages")
    traj_out = os.path.join(dynamic_root, "trajectories")
    epi_out = os.path.join(dynamic_root, "episodes")
    for d in (dynamic_root, scenes_out, stages_out, traj_out, epi_out):
        os.makedirs(d, exist_ok=True)

    for i, (scene, avatar_id) in enumerate(zip(scenes, avatars)):
        gender = genders[i] if genders else AVATAR_GENDER.get(avatar_id, "female")
        navmesh = os.path.join(gs, args.scene_split, scene, f"{scene}.navmesh")
        if not os.path.exists(navmesh):
            print(f"[SKIP] {scene}: navmesh not found: {navmesh}", file=sys.stderr)
            continue
        driver_pkl = os.path.join(traj_out, f"{scene}.driver.pkl")
        print(f"\n=== {scene}  (avatar{avatar_id}, gender={gender}) ===")

        if "trajectory" in args.stages:
            if os.path.exists(driver_pkl):
                print(f"  [trajectory] exists, skipping: {driver_pkl}")
            else:
                pf = load_pathfinder(navmesh)
                ref = navmesh_center_ref(pf, seed=args.seed)
                del pf
                print(f"  [trajectory] baking (path-ref={np.round(ref,3).tolist()}) ...")
                bake_trajectory(
                    navmesh_path=navmesh, out_pkl=driver_pkl, path_ref=ref,
                    gender=gender, gamma_root=args.gamma_root,
                    gamma_env=args.gamma_env, conda_bin=args.conda_bin, tool=args.traj_tool,
                    path_length=args.path_length, gamma_depth=args.gamma_depth, fps=args.fps,
                    gpu_index=args.gpu_index, seed=args.seed + avatar_id,
                    log_path=os.path.join(traj_out, f"{scene}.gen.log"),
                )
                print(f"  [trajectory] -> {driver_pkl}")

        if "wiring" in args.stages:
            stage_cfg = write_stage_config(
                scene=scene, gs_scenes_dir=gs, stages_out_dir=stages_out,
                scene_split=args.scene_split,
            )
            off = (args.offset_y if args.offset_y is not None
                   else compute_offset_y(driver_pkl, navmesh, args.scale))
            inst = write_scene_instance(
                scene=scene, avatar_id=avatar_id, gs_scenes_dir=gs,
                scenes_out_dir=scenes_out, driver_pkl=driver_pkl,
                offset_y=off, scale=args.scale,
            )
            print(f"  [wiring] -> {inst}  (+ {os.path.basename(stage_cfg)})")

        if "episodes" in args.stages:
            # scene_id is the scene_instance path relative to scenes_dir; the dataset loader
            # joins scenes_dir so it becomes the registered scene_instance handle in the
            # active dynamic_nav dataset (which supplies the stage + navmesh + gaussian_avatars).
            scene_id_rel = f"gs_scenes/dynamic_nav/scenes/{scene}.scene_instance.json"
            ds_cfg_rel = rel_to(
                root, os.path.join(dynamic_root, "dynamic_nav.scene_dataset_config.json")
            )
            split_counts = [
                (s, n) for s, n in
                (("train", args.train_episodes), ("val", args.val_episodes))
                if s in args.splits
            ]
            for split, n in split_counts:
                shard = generate_episodes(
                    scene=scene, navmesh_path=navmesh, driver_pkl=driver_pkl,
                    num_episodes=n, split=split, scene_id_rel=scene_id_rel,
                    scene_dataset_config=ds_cfg_rel, out_dir=epi_out,
                    rng_seed=args.seed + avatar_id + (0 if split == "train" else 10000),
                    spawn_min=args.spawn_min, spawn_max=args.spawn_max,
                    goal_min=args.goal_min, goal_max=args.goal_max,
                    success_radius=args.success_radius,
                )
                print(f"  [episodes:{split}] {n} -> {shard}")

    if "wiring" in args.stages or "episodes" in args.stages:
        for split in args.splits:
            write_split_index(out_dir=epi_out, split=split)
        cfg = write_dataset_config(dynamic_root=dynamic_root, gs_scenes_dir=gs,
                                   scenes=scenes, scene_split=args.scene_split)
        print(f"\n[dataset-config] -> {cfg}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
