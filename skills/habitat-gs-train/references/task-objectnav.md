# ObjectNav (Habitat-Lab + DDPPO)

Navigate to any instance of a target object category. Sensors: RGB, Depth, GPS, Compass, ObjectGoal. Actions: move_forward, turn_left, turn_right, look_up, look_down, stop. Policy: `PointNavResNetPolicy` (ResNet-50 + 2-layer LSTM). Env: `habitat-gs`.

## Step 1 — Generate episodes (the involved step)

ObjectNav has no ground-truth object labels on most scenes, so the default generator **detects objects with SAM + CLIP**. This step (only) needs the SAM + CLIP checkpoints from `prerequisites.md`.

```bash
conda activate habitat-gs
# outdoor categories (IDs 0–11) on outdoor scenes
python scripts_gs/generate_objectnav_episodes.py
# indoor categories (IDs 12–21) on InteriorGS scenes
python scripts_gs/generate_objectnav_episodes.py --indoor
```
Pipeline per scene: scan ~200 navigable points × 4 rotations → SAM masks → CLIP zero-shot classify → back-project masked depth to 3D → DBSCAN cluster per category → quality filters → sample episodes toward object view_points. A 22-class unified taxonomy is written to every file so indoor + outdoor episodes merge into one `ObjectNav-v1` dataset. Extra flags: `--num-sample-points`, `--visualize` (saves annotated detection PNGs), `--device`.

Categories:
- Outdoor (default, 0–11): car, bench, tree, street lamp, traffic sign, fire hydrant, trash can, bicycle, potted plant, barrier, statue, chair.
- Indoor (`--indoor`, 12–21): sofa, bed, dining table, toilet, sink, tv, refrigerator, bookshelf, cabinet, lamp.

Output: `episodes/objectnav/<split>/content/<scene>.json.gz` (with `goals_by_category`) + a `<split>.json.gz` index carrying the category maps.

**More accurate alternatives** (SAM+CLIP may mislocalize):
- InteriorGS scenes with GT boxes: `python scripts_gs/generate_objectnav_episodes_interiorgs.py` (uses `<id>_labels.json`; byte-identical output schema; writes to `episodes/objectnav_interiorgs_gt/` by default). `..._interiorgs_special.py` handles two furniture-less edge-case scenes.
- Any custom scene: the `web_tools/objectnav_helper` browser tool to hand-place + label objects.

## Step 2 — Train

```bash
bash scripts_gs/train_objectnav.sh --output output/objectnav    # default 2.5e9 steps
bash scripts_gs/train_objectnav.sh --output output/objectnav --num-envs 8 --num-gpus 4
```
Same options as the other RL tasks (`--output` required; `--num-envs/--num-gpus/--total-steps/--num-ckpts/--pretrained-ckpt`; extra args → Hydra overrides). Aborts if `episodes/objectnav/train/content` is missing.

## Step 3 — Evaluate

```bash
bash scripts_gs/eval_objectnav.sh --ckpt output/objectnav/checkpoints/ckpt.0.pth
```
Options identical to the other RL eval scripts (`--ckpt` single file recommended — see the directory caveat in `task-pointnav.md`; `--num-envs`, `--video-dir`).

## Config

`configs/ddppo_objectnav_gs_{train,eval}.yaml`. RGB+Depth 256×256, `max_episode_steps=500`, `ppo.num_steps=64`, `ppo.ppo_epoch=4`, reward overrides `success_reward=2.5`, `slack_reward=-1e-3`, dataset type `ObjectNav-v1`. Eval may log a missing `<scene>.gs.scn` semantic descriptor — harmless warning.
