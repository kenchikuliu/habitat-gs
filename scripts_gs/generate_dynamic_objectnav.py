#!/usr/bin/env python3
r"""Retarget the manually-annotated ObjectNav episodes onto the dynamic-nav scenes.

Takes the existing ObjectNav shards for scene01-10 (static GS scenes) and produces
`gs_scenes/dynamic_nav/episodes_objectnav/` where each episode:

  * points at the dynamic scene_instance (so the walking GS avatar is loaded):
    `scene_id = gs_scenes/dynamic_nav/scenes/<scene>.scene_instance.json` and
    `scene_dataset_config = .../dynamic_nav.scene_dataset_config.json`;
  * starts in an annulus 1.5-4 m around the avatar's trajectory start, facing the
    avatar (so agent and human must interact);
  * keeps the object annotations verbatim -- `goals_by_category` is carried over with
    its keys rewritten to match the new `scene_id` basename (habitat resolves goals
    via `basename(scene_id)_<category>`);
  * has `info.geodesic_distance` recomputed from the new start to the nearest goal
    view point, and is dropped if no view point is reachable within [1, 30] m.

Usage (habitat-gs conda env, from the repo root):
    python scripts_gs/generate_dynamic_objectnav.py --scenes scene01 scene02
    python scripts_gs/generate_dynamic_objectnav.py --scenes scene01 --train-episodes 20
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dynamic_nav import (  # noqa: E402
    _sample_start_near,
    _yaw_quat_facing,
    driver_start_xyz,
    geodesic,
    load_pathfinder,
    rel_to,
    repo_root,
)


def _goal_view_positions(goals: list) -> np.ndarray:
    pts = []
    for g in goals:
        for vp in g.get("view_points", []):
            pos = vp.get("agent_state", {}).get("position") if isinstance(vp, dict) else None
            if pos is not None:
                pts.append(pos)
    return np.asarray(pts, dtype=np.float32)


def retarget_scene(
    *, scene: str, src_shard: str, out_dir: str, gs: str, root: str,
    n_train: int, n_val: int, spawn_min: float, spawn_max: float,
    geo_min: float, geo_max: float, seed: int, scene_split: str = "train",
) -> tuple[dict, dict]:
    with gzip.open(src_shard, "rt") as f:
        src = json.load(f)

    old_base = os.path.basename(src["episodes"][0]["scene_id"])  # e.g. scene01.gs.ply
    new_scene_id = f"gs_scenes/dynamic_nav/scenes/{scene}.scene_instance.json"
    new_base = os.path.basename(new_scene_id)
    ds_cfg = rel_to(root, os.path.join(gs, "dynamic_nav/dynamic_nav.scene_dataset_config.json"))

    # goals_by_category with rewritten keys (habitat: basename(scene_id)_category)
    gbc = {
        k.replace(old_base, new_base, 1): v for k, v in src["goals_by_category"].items()
    }

    navmesh = os.path.join(gs, scene_split, scene, f"{scene}.navmesh")
    pf = load_pathfinder(navmesh)
    driver = os.path.join(gs, "dynamic_nav/trajectories", f"{scene}.driver.pkl")
    avatar_start = driver_start_xyz(driver)
    rng = np.random.RandomState(seed)

    episodes = []
    dropped = 0
    order = rng.permutation(len(src["episodes"]))
    for idx in order:
        if len(episodes) >= n_train + n_val:
            break
        ep = copy.deepcopy(src["episodes"][int(idx)])
        key = f"{new_base}_{ep['object_category']}"
        goals = gbc.get(key)
        if not goals:
            dropped += 1
            continue
        vps = _goal_view_positions(goals)
        if vps.size == 0:
            dropped += 1
            continue
        start = _sample_start_near(
            pf, avatar_start, rmin=spawn_min, rmax=spawn_max, rng=rng
        )
        if start is None:
            dropped += 1
            continue
        geos = [geodesic(pf, start, vp) for vp in vps]
        geos = [g for g in geos if np.isfinite(g)]
        if not geos or min(geos) < geo_min or min(geos) > geo_max:
            dropped += 1
            continue
        ep["scene_id"] = new_scene_id
        ep["scene_dataset_config"] = ds_cfg
        ep["start_position"] = [float(x) for x in start]
        ep["start_rotation"] = _yaw_quat_facing(start[[0, 2]], avatar_start[[0, 2]])
        ep["info"] = dict(ep.get("info") or {})
        ep["info"]["geodesic_distance"] = float(min(geos))
        ep["goals"] = []
        episodes.append(ep)

    if len(episodes) < n_train + n_val:
        print(f"  [WARN] {scene}: only {len(episodes)}/{n_train + n_val} episodes "
              f"(dropped {dropped})", file=sys.stderr)

    for i, ep in enumerate(episodes):
        ep["episode_id"] = str(i)

    maps = {
        "category_to_task_category_id": src["category_to_task_category_id"],
        "category_to_scene_annotation_category_id": src[
            "category_to_scene_annotation_category_id"
        ],
    }
    train_shard = {**maps, "goals_by_category": gbc, "episodes": episodes[:n_train]}
    val_eps = episodes[n_train:n_train + n_val]
    for i, ep in enumerate(val_eps):
        ep["episode_id"] = str(i)
    val_shard = {**maps, "goals_by_category": gbc, "episodes": val_eps}

    for split, shard in (("train", train_shard), ("val", val_shard)):
        cdir = os.path.join(out_dir, split, "content")
        os.makedirs(cdir, exist_ok=True)
        with gzip.open(os.path.join(cdir, f"{scene}.json.gz"), "wt") as f:
            json.dump(shard, f)
    print(f"  {scene}: {len(train_shard['episodes'])} train + {len(val_shard['episodes'])} val "
          f"(dropped {dropped}); goals keys -> '{new_base}_<cat>'")
    return maps, {"train": len(train_shard["episodes"]), "val": len(val_shard["episodes"])}


def main() -> int:
    root = repo_root()
    os.chdir(root)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="scene names, e.g. scene01 scene05")
    ap.add_argument("--scene-split", default="train", choices=["train", "val"],
                    help="gs_scenes subfolder the scene assets live in (default: train)")
    ap.add_argument("--gs-scenes", default=os.path.join(root, "data/scene_datasets/gs_scenes"))
    ap.add_argument("--train-episodes", type=int, default=100)
    ap.add_argument("--val-episodes", type=int, default=10)
    ap.add_argument("--spawn-min", type=float, default=1.5)
    ap.add_argument("--spawn-max", type=float, default=4.0)
    ap.add_argument("--geo-min", type=float, default=1.0)
    ap.add_argument("--geo-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenes = list(args.scenes)
    gs = args.gs_scenes
    out_dir = os.path.join(gs, "dynamic_nav/episodes_objectnav")
    maps = None
    for scene in scenes:
        src = os.path.join(gs, "episodes/objectnav", args.scene_split, "content", f"{scene}.json.gz")
        if not os.path.exists(src):
            print(f"  [SKIP] {scene}: no source shard {src}", file=sys.stderr)
            continue
        maps, _ = retarget_scene(
            scene=scene, src_shard=src, out_dir=out_dir, gs=gs, root=root,
            n_train=args.train_episodes, n_val=args.val_episodes,
            spawn_min=args.spawn_min, spawn_max=args.spawn_max,
            geo_min=args.geo_min, geo_max=args.geo_max, seed=args.seed,
            scene_split=args.scene_split,
        )

    if maps:
        # top-level split indices must carry the category maps (dataset reads them
        # from the main file before loading the per-scene content shards).
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_dir, split), exist_ok=True)
            with gzip.open(os.path.join(out_dir, split, f"{split}.json.gz"), "wt") as f:
                json.dump({**maps, "goals_by_category": {}, "episodes": []}, f)
        print(f"[index] wrote {out_dir}/{{train,val}}/*.json.gz")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
