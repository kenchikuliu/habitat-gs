#!/usr/bin/env python3
"""objectnav_helper — browser tool to annotate objects on a 3D Gaussian Splatting
scene and generate Habitat ObjectNav episodes. A sibling of navmesh_editor.

Custom-built (non-InteriorGS) scenes have no ground-truth object annotations, and the
automatic SAM+CLIP pipeline mis-localizes objects badly. This tool lets you place the
real objects by hand on the rendered GS scene, label each with an ObjectNav category,
and one-click generate correct ObjectNav episodes (view_points sampled on the scene's
navmesh, episodes sampled exactly like the rest of the dataset).

Run with the habitat-gs conda env (needs habitat_sim + flask + plyfile):

    conda activate habitat-gs
    python web_tools/objectnav_helper/server.py --port 8081
    # then open http://localhost:8081  (forward the port if running remotely)

The scene directory / episode output / episode counts can be overridden via flags
(see --help). Annotating happens directly in the habitat (Y-up) frame, so no
coordinate transform is needed.
"""
import argparse
import base64
import contextlib
import glob
import gzip
import importlib.util
import json
import logging
import os
import re
import threading

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

# Silence habitat_sim's verbose C++ logging (e.g. the "getRandomNavigablePointInCircle: Failed"
# retry spam emitted during view_point sampling) — set before habitat_sim is ever imported.
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("MAGNUM_LOG", "quiet")


@contextlib.contextmanager
def quiet_native_stderr():
    """Temporarily redirect the process's C-level stderr (fd 2) to /dev/null, so habitat_sim's
    native retry warnings during navmesh sampling don't flood the terminal. Python exceptions are
    unaffected (they don't go through fd 2)."""
    import sys
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root is two levels up: web_tools/objectnav_helper/ -> web_tools/ -> repo root.
REPO_ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
STATIC = os.path.join(HERE, "static")
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)

# Scene directory (same convention as navmesh_editor): a folder whose subdirs hold GS
# scenes, either directly (.../gs_scenes/train) or grouped one level deeper (gs_scenes
# with train/, val/). Override with --gs-dir / the env var.
GS_DIR = os.environ.get(
    "OBJECTNAV_HELPER_GS_DIR",
    os.path.join(REPO_ROOT, "data", "scene_datasets", "gs_scenes"),
)
# Where generated episodes (+ a re-editable annotations sidecar) are written. Defaults
# to the canonical ObjectNav location so saving a scene replaces its episode file.
OUT_DIR = os.environ.get(
    "OBJECTNAV_HELPER_OUT_DIR",
    os.path.join(REPO_ROOT, "data", "scene_datasets", "gs_scenes", "episodes", "objectnav"),
)

# Reuse the ObjectNav episode-generation logic (view_point sampling, episode sampler,
# 22-class taxonomy, writer) from the InteriorGS generator — single source of truth.
_GEN_PATH = os.path.join(REPO_ROOT, "scripts_gs", "generate_objectnav_episodes_interiorgs.py")
_spec = importlib.util.spec_from_file_location("objnav_gen", _GEN_PATH)
GEN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GEN)
CATEGORIES = list(GEN.UNIFIED_CATEGORY_TO_ID.keys())   # the 22 ObjectNav classes

SH_C0 = 0.28209479177387814
SPLAT_ALPHA_MIN = 0.1
MAX_SPLATS = 1_000_000
DEFAULT_OBJ_RADIUS = 0.75       # footprint radius used to sample view_points around a placed object
DEFAULT_FURTHEST = 30.0         # max episode geodesic distance (m); outdoor scenes are large

logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = Flask(__name__, static_folder=None)
_lock = threading.Lock()
_scene_cache = {}     # ply_path -> meta dict
_pf_cache = {}        # navmesh_path -> (PathFinder, floor_y)


# --------------------------------------------------------------------------- #
#  scene discovery / resolution  (same convention as navmesh_editor)
# --------------------------------------------------------------------------- #
def scene_dir_and_split(ply_path):
    d = os.path.dirname(ply_path)
    name = os.path.basename(d)
    group = os.path.basename(os.path.dirname(d))   # train / val / ...
    return d, name, group


def ann_path(scene):
    return os.path.join(OUT_DIR, "_annotations", f"{scene}.json")


def load_existing_annotations(scene, split):
    """Re-editable annotations for a scene, with two sources (sidecar wins):
      1. the tool's own sidecar  <out>/_annotations/<scene>.json  (has per-object radius)
      2. fallback: the scene's existing ObjectNav episode file — reconstruct one marker
         per goal from its center `position` + `object_category` (so interior scenes and
         even the old auto-generated outdoor episodes open pre-populated and editable).
    Returns (annotations_dict_or_None, source_str)."""
    if os.path.exists(ann_path(scene)):
        try:
            return json.load(open(ann_path(scene))), "sidecar"
        except Exception:
            pass
    ep = os.path.join(OUT_DIR, split, "content", f"{scene}.json.gz")
    if os.path.exists(ep):
        try:
            d = json.load(gzip.open(ep, "rt"))
            objs = [{"category": g["object_category"], "position": g["position"]}
                    for goals in d.get("goals_by_category", {}).values() for g in goals]
            if objs:
                return {"scene": scene, "split": split, "objects": objs}, "episodes"
        except Exception:
            pass
    return None, None


def list_scenes():
    out, seen = [], set()
    for sub in sorted(glob.glob(os.path.join(GS_DIR, "*", ""))):
        plys = (sorted(glob.glob(os.path.join(sub, "*.gs.ply")))
                or sorted(glob.glob(os.path.join(sub, "*", "*.gs.ply"))))
        for ply in plys:
            if ply in seen:
                continue
            seen.add(ply)
            d, name, group = scene_dir_and_split(ply)
            ep = os.path.join(OUT_DIR, group, "content", name + ".json.gz")
            out.append({"name": name, "split": group,
                        "has_navmesh": os.path.exists(os.path.join(d, name + ".navmesh")),
                        "has_annotations": os.path.exists(ann_path(name)) or os.path.exists(ep)})
    return out


def resolve_ply(scene):
    if not scene:
        return None
    if scene.endswith(".gs.ply") and os.path.isfile(scene):
        return os.path.abspath(scene)
    for cand in (scene, os.path.join(GS_DIR, scene)):
        if os.path.isdir(cand):
            hits = sorted(glob.glob(os.path.join(cand, "*.gs.ply")))
            if hits:
                return hits[0]
    hits = sorted(glob.glob(os.path.join(GS_DIR, "*", scene, "*.gs.ply")))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
#  read .gs.ply ONCE -> write .splat for the browser + compute floor/ceiling
# --------------------------------------------------------------------------- #
def floor_ceiling_estimate(y):
    """Floor = densest layer in the lower half of the height range; ceiling = densest
    upper-half layer a plausible room-height above (else None for open/outdoor)."""
    ylo, yhi = np.percentile(y, [1, 99])
    if yhi - ylo < 1e-3:
        return float(ylo), None
    nb = max(40, int((yhi - ylo) / 0.05))
    h, edges = np.histogram(y, bins=nb, range=(ylo, yhi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    low = centers < 0.5 * (ylo + yhi)
    floor = float(centers[low][int(np.argmax(h[low]))])
    floor_n = int(h[low].max())
    ceiling = None
    hi_idx = np.nonzero(~low)[0]
    if len(hi_idx):
        j = int(hi_idx[int(np.argmax(h[hi_idx]))])
        if 1.2 <= float(centers[j]) - floor <= 10.0 and h[j] >= 0.3 * floor_n:
            ceiling = float(centers[j])
    return floor, ceiling


def prepare_scene(ply_path):
    """Idempotent: ensure _tmp/<scene>.splat exists; return scene metadata."""
    if ply_path in _scene_cache and os.path.exists(_scene_cache[ply_path]["splat"]):
        return _scene_cache[ply_path]
    import time
    from plyfile import PlyData
    t0 = time.time()
    v = PlyData.read(ply_path)["vertex"]
    n = len(v)
    names = v.data.dtype.names or ()
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32, copy=False)
    a = (1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float32)))) if "opacity" in names else np.ones(n, np.float32)
    idx = np.nonzero(a > SPLAT_ALPHA_MIN)[0]
    if len(idx) > MAX_SPLATS:
        idx = idx[np.argpartition(a[idx], -MAX_SPLATS)[-MAX_SPLATS:]]
    dc = np.column_stack([v["f_dc_0"][idx], v["f_dc_1"][idx], v["f_dc_2"][idx]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    scale = np.exp(np.column_stack([v["scale_0"][idx], v["scale_1"][idx], v["scale_2"][idx]]).astype(np.float32))
    q = np.column_stack([v[f"rot_{i}"][idx] for i in range(4)]).astype(np.float32)
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

    _, name, split = scene_dir_and_split(ply_path)
    splat = os.path.join(TMP, f"{name}.splat")
    out = np.zeros(len(idx), dtype=[('pos', '<f4', 3), ('scale', '<f4', 3),
                                    ('rgba', 'u1', 4), ('rot', 'u1', 4)])
    out['pos'] = xyz[idx]
    out['scale'] = scale
    out['rgba'][:, :3] = (rgb * 255).astype(np.uint8)
    out['rgba'][:, 3] = (a[idx] * 255).astype(np.uint8)
    out['rot'] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
    out.tofile(splat)
    print(f"[splat] {name}: {n:,} -> {len(idx):,} splats "
          f"({os.path.getsize(splat)/1024**2:.1f} MB) in {time.time()-t0:.1f}s", flush=True)

    fy, ceiling_y = floor_ceiling_estimate(xyz[idx, 1])
    meta = {"name": name, "split": split, "splat": splat,
            "n_splats": int(len(idx)), "n_total": int(n),
            "bounds": {"min": xyz.min(0).tolist(), "max": xyz.max(0).tolist()},
            "floor_y": fy, "ceiling_y": ceiling_y}
    _scene_cache[ply_path] = meta
    return meta


# --------------------------------------------------------------------------- #
#  annotations -> ObjectNav episodes (reuses GEN.sample_view_points / episodes / writer)
# --------------------------------------------------------------------------- #
def _pathfinder(navmesh_path):
    if navmesh_path in _pf_cache:
        return _pf_cache[navmesh_path]
    import habitat_sim
    pf = habitat_sim.nav.PathFinder()
    if not pf.load_nav_mesh(navmesh_path) or not pf.is_loaded:
        return None, None
    ys = [pf.get_random_navigable_point()[1] for _ in range(60)]
    floor_y = float(np.median([y for y in ys if not np.isnan(y)]))
    _pf_cache[navmesh_path] = (pf, floor_y)
    return pf, floor_y


def navmesh_payload(navmesh_path):
    """Triangles of the scene's navmesh, for read-only display in the browser."""
    if not os.path.exists(navmesh_path):
        return None
    with quiet_native_stderr():
        pf, _ = _pathfinder(navmesh_path)
        if pf is None:
            return None
        try:
            tris = np.asarray(pf.build_navmesh_vertices(-1), np.float32).reshape(-1, 3)
        except Exception:
            return None
    return {"tris_b64": base64.b64encode(tris.astype("<f4").tobytes()).decode(),
            "n_tris": int(len(tris) // 3),
            "navigable_area": float(getattr(pf, "navigable_area", 0.0))}


def generate(ply_path, objects, params):
    """objects: [{category, position:[x,y,z], radius?}] -> write ObjectNav episode file."""
    sdir, name, split = scene_dir_and_split(ply_path)
    navmesh = os.path.join(sdir, name + ".navmesh")
    if not os.path.exists(navmesh):
        return {"ok": False, "error": f"no navmesh for {name} — build one with navmesh_editor first"}
    pf, floor_y = _pathfinder(navmesh)
    if pf is None:
        return {"ok": False, "error": "failed to load navmesh"}

    radius = float(params.get("radius", DEFAULT_OBJ_RADIUS))
    n_ep = int(params.get("num_train", GEN.NUM_TRAIN_EPISODES)) if split == "train" \
        else int(params.get("num_val", GEN.NUM_VAL_EPISODES))
    GEN.FURTHEST_DIST = float(params.get("furthest", DEFAULT_FURTHEST))   # episodes can span large scenes
    seed = int(params.get("seed", 42))
    pf.seed(seed)   # seed BEFORE view_point sampling so re-saving identical annotations is reproducible
                    # (GEN.generate_episodes re-seeds pf + numpy for the episode-sampling stage)

    objects_per_category = {}
    dropped = []
    # view_point + episode sampling can flood the terminal with native "Failed to
    # getRandomNavigablePoint" retry lines for objects far from the navmesh — silence that.
    with quiet_native_stderr():
        for i, o in enumerate(objects):
            cat = o.get("category")
            if cat not in GEN.UNIFIED_CATEGORY_TO_ID:
                dropped.append({"i": i, "category": cat, "why": "unknown category"}); continue
            center = np.array(o["position"], dtype=np.float64)
            foot_r = float(o.get("radius", radius))
            vps = GEN.sample_view_points(pf, center, foot_r, floor_y)
            if len(vps) < GEN.MIN_VIEWPOINTS_PER_OBJECT:
                dropped.append({"i": i, "category": cat, "why": "too far from walkable area (no navmesh stance)"}); continue
            objects_per_category.setdefault(cat, []).append(
                {"position": [float(center[0]), float(center[1]), float(center[2])], "view_points": vps})

        if not objects_per_category:
            return {"ok": False,
                    "error": (f"None of the {len(objects)} objects is near the walkable area (navmesh). "
                              f"Move objects onto/near the green navmesh, or increase 'obj radius'."),
                    "n_objects": len(objects), "n_kept": 0, "n_dropped": len(dropped), "dropped": dropped}

        scene_basename = f"{name}.gs.ply"
        gbc = GEN.build_goals_by_category(objects_per_category, scene_basename)
        scene_id = f"gs_scenes/{split}/{name}/{name}.gs.ply"
        episodes = GEN.generate_episodes(pf, scene_id, gbc, n_ep, seed)

    out_file = os.path.join(OUT_DIR, split, "content", f"{name}.json.gz")
    GEN.save_dataset(episodes, gbc, out_file)

    # re-editable annotations sidecar
    os.makedirs(os.path.dirname(ann_path(name)), exist_ok=True)
    with open(ann_path(name), "w") as f:
        json.dump({"scene": name, "split": split, "radius": radius,
                   "objects": objects}, f, indent=1)

    cats = {c: len(v) for c, v in objects_per_category.items()}
    return {"ok": True, "path": out_file, "split": split,
            "n_objects": len(objects), "n_kept": sum(cats.values()),
            "n_dropped": len(dropped), "dropped": dropped,
            "categories": cats, "n_episodes": len(episodes), "target_episodes": n_ep}


# --------------------------------------------------------------------------- #
#  HTTP
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC, p)


@app.route("/api/scenes")
def api_scenes():
    return jsonify({"scenes": list_scenes(), "default": app.config["DEFAULT_SCENE"],
                    "categories": CATEGORIES})


@app.route("/api/scene")
def api_scene():
    name = request.args.get("name", "")
    ply = resolve_ply(name)
    if not ply:
        return jsonify({"error": f"scene not found: {name}"}), 404
    meta = prepare_scene(ply)
    app.config["CUR_PLY"][name] = ply
    sd, sname, split = scene_dir_and_split(ply)
    existing, ann_source = load_existing_annotations(sname, split)
    navmesh = navmesh_payload(os.path.join(sd, sname + ".navmesh"))
    return jsonify({
        "name": name, "split": split,
        "splat_url": f"/api/splat?name={name}",
        "n_splats": meta["n_splats"], "n_total": meta["n_total"],
        "bounds": meta["bounds"], "floor_y": meta["floor_y"], "ceiling_y": meta.get("ceiling_y"),
        "has_navmesh": navmesh is not None,
        "navmesh": navmesh,
        "categories": CATEGORIES, "annotations": existing, "annotations_source": ann_source,
    })


@app.route("/api/splat")
def api_splat():
    name = request.args.get("name", "")
    ply = app.config["CUR_PLY"].get(name) or resolve_ply(name)
    if not ply:
        return jsonify({"error": "scene not found"}), 404
    return send_file(prepare_scene(ply)["splat"], mimetype="application/octet-stream",
                     as_attachment=False, conditional=True)


@app.route("/api/save", methods=["POST"])
def api_save():
    body = request.get_json(silent=True) or {}
    name = body.get("scene")
    ply = (app.config["CUR_PLY"].get(name) or resolve_ply(name)) if name else None
    if not ply:
        return jsonify({"ok": False, "error": "scene not loaded"}), 400
    objs = body.get("objects", [])
    if not objs:
        return jsonify({"ok": False, "error": "no objects placed — annotate some objects first"}), 400
    with _lock:
        return jsonify(generate(ply, objs, body.get("params", {})))


def main():
    global GS_DIR, OUT_DIR
    ap = argparse.ArgumentParser(
        description="objectnav_helper — annotate objects on a 3DGS scene and generate ObjectNav episodes")
    ap.add_argument("--scene", default=None, help="scene name/dir to open first")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--gs-dir", default=GS_DIR,
                    help="scene directory: a folder whose subdirs hold GS scenes")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="where ObjectNav episodes (+ annotations sidecar) are written")
    args = ap.parse_args()
    GS_DIR = os.path.abspath(args.gs_dir)
    OUT_DIR = os.path.abspath(args.out_dir)

    scenes = list_scenes()
    default = args.scene or (scenes[0]["name"] if scenes else None)
    app.config["DEFAULT_SCENE"] = default
    app.config["CUR_PLY"] = {}
    print(f"objectnav_helper: {len(scenes)} scenes  default={default}")
    print(f"  gs_dir : {GS_DIR}")
    print(f"  out_dir: {OUT_DIR}")
    print(f"  open  http://{args.host}:{args.port}")
    print(f"  (ssh -L {args.port}:localhost:{args.port} <server>  then open in your browser)")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
