# PointNav (Habitat-Lab + DDPPO)

Navigate to a goal given as relative GPS coordinates. Sensors: RGB, Depth, GPS, Compass. Actions: move_forward, turn_left, turn_right, stop. Policy: `PointNavResNetPolicy` (ResNet-50 + 2-layer LSTM). Env: `habitat-gs`.

## Step 1 — Generate episodes (optional; released dataset ships these)

```bash
conda activate habitat-gs
python scripts_gs/generate_pointnav_episodes.py
```
Defaults: 1000 train / 100 val episodes per scene, seed 42. Uses a pathfinder-only sim (loads each scene `.navmesh`), samples navigable start/goal pairs, filters by geodesic distance (1.0–30.0 m), same-floor, and a geodesic/euclidean straightness ratio. Writes `episodes/pointnav/<split>/content/<scene>.json.gz` + an empty `<split>.json.gz` index.

Useful flags (shared by all three RL generators): `--scenes-root`, `--output-root`, `--train-episodes N`, `--val-episodes N`, `--seed`, `--scene <name>` (single scene), `--scenes a,b,c` (subset in one process).

## Step 2 — Train

```bash
bash scripts_gs/train_pointnav.sh --output output/pointnav          # default 5e8 steps
bash scripts_gs/train_pointnav.sh --output output/pointnav --num-envs 8 --num-gpus 2
```
Options (all `train_*.sh` share them):
```
--output DIR             output dir for checkpoints + tensorboard (REQUIRED)
--num-envs N             parallel envs per GPU (default 4; for TRAINING must be >= ppo.num_mini_batch = 2)
--num-gpus N             GPUs for DDPPO (default 1; >1 uses torch.distributed.launch)
--total-steps N          total training steps (PointNav default 5e8)
--num-ckpts N            checkpoints to save (default 100)
--pretrained-ckpt PATH   fine-tune from a .pth (sets ddppo.pretrained=True, re-inits critic)
```
Extra args are forwarded verbatim as Hydra overrides. The script aborts if `episodes/pointnav/train/content` is missing. Output: `output/pointnav/{checkpoints/, tb/, train.log}`.

For resume / fine-tune / encoder-transfer details see `training-and-finetuning.md`.

## Step 3 — Evaluate

```bash
# Recommended: evaluate a SINGLE checkpoint file (clean, verified)
bash scripts_gs/eval_pointnav.sh --ckpt output/pointnav/checkpoints/ckpt.0.pth
bash scripts_gs/eval_pointnav.sh --ckpt <ckpt> --video-dir output/pointnav/videos
# Quick smoke on a few episodes:
bash scripts_gs/eval_pointnav.sh --ckpt <ckpt> habitat_baselines.test_episode_count=3
# Evaluate several existing checkpoints — loop over files (see caveat):
for c in output/pointnav/checkpoints/ckpt.*.pth; do bash scripts_gs/eval_pointnav.sh --ckpt "$c"; done
```
Options: `--ckpt PATH` (file or directory; REQUIRED), `--num-envs N` (default 1), `--video-dir DIR` (records rollout videos). Uses `--config-name=ddppo_pointnav_gs_eval` (`evaluate=True`, `num_environments=1`). Metrics print to stdout / the run's `eval.stdout` — see `outputs-and-metrics.md`.

> **Directory `--ckpt` caveat (verified by running it).** Passing a *directory* puts habitat-baselines into a polling "watcher" mode, NOT a one-shot "eval all". Two consequences: (1) after each checkpoint it writes an eval resume-state to `habitat_baselines.checkpoint_folder` (default `data/checkpoints`) — if that dir doesn't exist it crashes with `RuntimeError: Parent directory data/checkpoints does not exist`; (2) it keeps polling until it has evaluated `num_checkpoints` (default 100) checkpoints, counting only `ckpt.<N>.pth` files (`latest.pth` is ignored), so on a fixed folder it **hangs** waiting for more. To evaluate a fixed set reliably, loop over single files (above), or pass `habitat_baselines.checkpoint_folder=<existing dir> habitat_baselines.num_checkpoints=<count of ckpt.N.pth files>`.

## Config

`data/scene_datasets/gs_scenes/configs/ddppo_pointnav_gs_{train,eval}.yaml`. RGB+Depth 256×256, `max_episode_steps=500`, `ppo.num_steps=128`, dataset type `PointNav-v1`.
