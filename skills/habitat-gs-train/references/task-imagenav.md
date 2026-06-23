# ImageNav (Habitat-Lab + DDPPO)

Navigate to the viewpoint shown in a goal image. Sensors: RGB, Depth, ImageGoal. Actions: move_forward, turn_left, turn_right, stop. Policy: `PointNavResNetPolicy` (ResNet-50 + 2-layer LSTM). Env: `habitat-gs`. Same pipeline shape as PointNav; the differences are the goal sensor and a goal-image quality gate.

## Step 1 — Generate episodes (optional)

```bash
conda activate habitat-gs
python scripts_gs/generate_imagenav_episodes.py
```
Defaults: 1000 train / 100 val per scene, seed 12345. Builds a **rendering** sim (256×256 RGB) per scene. For each candidate goal it reproduces the `ImageGoalSensor`'s deterministic goal pose and renders it, then a **quality gate** rejects featureless goals (kept iff laplacian_std ≥ 8.0 OR rgb_std ≥ 15.0) so sky/blank-wall goals are filtered out. The goal image itself is NOT stored — it is rendered on the fly during training. Episode schema is identical to PointNav. Writes `episodes/imagenav/<split>/{content/<scene>.json.gz, <split>.json.gz}`.

Shared flags: `--scenes-root`, `--output-root`, `--train-episodes`, `--val-episodes`, `--seed`, `--scene`, `--scenes`.

## Step 2 — Train

```bash
bash scripts_gs/train_imagenav.sh --output output/imagenav    # default 2.5e9 steps
```
Same options as PointNav (`--output` required; `--num-envs`, `--num-gpus`, `--total-steps`, `--num-ckpts`, `--pretrained-ckpt`; extra args → Hydra overrides). Aborts if `episodes/imagenav/train/content` is missing. Output: `output/imagenav/{checkpoints/, tb/, train.log}`.

Common transfer use case — initialize the visual encoder from a trained PointNav policy:
```bash
bash scripts_gs/train_imagenav.sh \
    --output output/imagenav_ft \
    --pretrained-ckpt output/pointnav/checkpoints/ckpt.99.pth \
    habitat_baselines.rl.ddppo.pretrained=False \
    habitat_baselines.rl.ddppo.pretrained_encoder=True \
    habitat_baselines.rl.ddppo.train_encoder=False   # freeze encoder
```
See `training-and-finetuning.md`.

## Step 3 — Evaluate

```bash
bash scripts_gs/eval_imagenav.sh --ckpt output/imagenav/checkpoints/ckpt.0.pth
```
Options identical to PointNav eval (`--ckpt` single file recommended — see the directory caveat in `task-pointnav.md`; `--num-envs`, `--video-dir`). Note: `run_imagenav.py` automatically removes the `top_down_map` measurement (GS scenes have no overhead mesh for it) — verified: ImageNav eval runs cleanly because of this.

## Config

`configs/ddppo_imagenav_gs_{train,eval}.yaml`. RGB+Depth 256×256, `max_episode_steps=1000`, `ppo.num_steps=64`, task reward overrides `success_reward=2.5`, `slack_reward=-1e-3`, dataset type `PointNav-v1`.
