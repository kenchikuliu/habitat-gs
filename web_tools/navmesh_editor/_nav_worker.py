#!/usr/bin/env python3
"""Navmesh worker for navmesh_editor — run with a Python that has habitat_sim.

server.py launches this once as `python _nav_worker.py serve`. It imports
habitat_sim a single time and then handles JSON requests from stdin (one per
line), replying with a single line that starts with `RESULT:` (habitat_sim's own
stdout/stderr logs are ignored by the server). Keeping habitat_sim in this
subprocess lets server.py run under any Python, and amortises the import so every
bake after the first is fast.

Requests:
  {"cmd":"read", "path": "<navmesh>"}              -> stats + triangles of a navmesh
  {"cmd":"bake", "obj": "<obj>", "out": "<navmesh>", "params": {...}}
                                                   -> build a navmesh from a proxy OBJ
  {"cmd":"ping"}                                   -> readiness check
"""
import sys
import json
import base64

import numpy as np


def payload(pf):
    tris = np.asarray(pf.build_navmesh_vertices(-1), np.float32).reshape(-1, 3)
    bmin, bmax = pf.get_bounds()
    return {"ok": True,
            "tris_b64": base64.b64encode(tris.tobytes()).decode(),
            "n_tris": int(len(tris) // 3),
            "navigable_area": float(pf.navigable_area),
            "num_islands": int(pf.num_islands),
            "bounds": {"min": [float(x) for x in bmin],
                       "max": [float(x) for x in bmax]}}


def emit(obj):
    # Leading newline so RESULT: always starts a fresh line, even if habitat_sim's
    # C++ layer left an unterminated line on stdout just before this.
    sys.stdout.write("\nRESULT:" + json.dumps(obj) + "\n")
    sys.stdout.flush()


def _settings(habitat_sim, pj):
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = float(pj.get("agent_radius", 0.1))
    ns.agent_height = float(pj.get("agent_height", 1.5))
    ns.agent_max_climb = float(pj.get("agent_max_climb", 0.2))
    ns.agent_max_slope = float(pj.get("agent_max_slope", 45.0))
    ns.cell_size = float(pj.get("cell_size", 0.05))
    ns.cell_height = float(pj.get("cell_height", 0.05))
    if "edge_max_error" in pj:
        ns.edge_max_error = float(pj["edge_max_error"])
    return ns


def serve():
    """Import habitat_sim once, then handle JSON requests from stdin until EOF."""
    try:
        import habitat_sim
    except Exception as e:
        emit({"ok": False, "error": f"habitat_sim import failed: {e}"})
        return
    sim = None
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    emit({"ok": True, "ready": True})                      # signal the server we're up
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            cmd = req.get("cmd")
            if cmd == "read":
                pf = habitat_sim.nav.PathFinder()
                if not pf.load_nav_mesh(req["path"]) or not pf.is_loaded:
                    emit({"ok": False, "error": "navmesh load failed"}); continue
                emit(payload(pf)); continue
            if cmd == "bake":
                ns = _settings(habitat_sim, req["params"])
                sc = habitat_sim.SimulatorConfiguration()
                sc.scene_id = req["obj"]; sc.create_renderer = False; sc.load_semantic_mesh = False
                cfg = habitat_sim.Configuration(sc, [agent_cfg])
                # FRESH Simulator each bake: habitat's ResourceManager caches assets
                # by filepath, so reusing the sim would re-bake the STALE cached mesh
                # and ignore edits. A new sim = a new, empty cache.
                if sim is not None:
                    try:
                        sim.close()
                    except Exception:
                        pass
                    sim = None
                sim = habitat_sim.Simulator(cfg)
                ok = sim.recompute_navmesh(sim.pathfinder, ns)
                if not ok or not sim.pathfinder.is_loaded:
                    emit({"ok": False, "error": "recompute_navmesh failed (try a different region / params)"}); continue
                sim.pathfinder.save_nav_mesh(req["out"])
                emit(payload(sim.pathfinder)); continue
            if cmd == "ping":
                emit({"ok": True, "pong": True}); continue
            emit({"ok": False, "error": f"unknown cmd: {cmd!r}"})
        except Exception as e:
            import traceback
            emit({"ok": False, "error": str(e), "trace": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "serve":
        sys.exit(f"usage: {sys.argv[0]} serve  (reads requests from stdin)")
    serve()
