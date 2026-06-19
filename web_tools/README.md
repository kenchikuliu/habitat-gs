# Habitat-GS Web Tools

Two browser tools to prepare GS scenes for navigation. Each runs a small local Flask server. Scene dir defaults to `data/scene_datasets/gs_scenes` (`--gs-dir` to override).

## navmesh_editor

Draw/edit a walkable area on the scene and bake a Habitat `.navmesh` for navigation:

```bash
conda activate habitat-gs
python web_tools/navmesh_editor/server.py --port 8080   # open http://localhost:8080
```

## objectnav_helper

Place/label objects on the scene and generate episodes for ObjectNav tasks:

```bash
conda activate habitat-gs
python web_tools/objectnav_helper/server.py --port 8081   # open http://localhost:8081
```
