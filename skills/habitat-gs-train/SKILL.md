---
name: habitat-gs-train
description: Train and evaluate a navigation policy in the habitat-gs simulator. Covers the full generate-episodes → train → evaluate flow for PointNav / ImageNav / ObjectNav (Habitat-Lab + DDPPO reinforcement learning) and for Vision-and-Language Navigation (StreamVLN, Uni-NaVid). Use when the user wants to train, fine-tune, resume, or evaluate a nav policy / agent on GS scenes — NOT for interactively piloting a live sim (use the habitat-gs-control skill for that).
---

# habitat-gs-train

Train and evaluate navigation policies on photo-realistic 3D Gaussian Splatting scenes in habitat-gs. Five tasks are supported, each with a **one-click** generate → train → evaluate pipeline driven by scripts under `scripts_gs/`.

**IMPORTANT — run from the repo root.** Every command below assumes the current directory is the `habitat-gs/` project root and that the right conda env is active. The scripts `cd` to the project root themselves, but paths like `data/scene_datasets/gs_scenes/...` are repo-relative. Never invent flags — the exact flags each script accepts are in `references/`.

## Pick the task first

| Task | Kind | Conda env | Backbone | Reference |
|---|---|---|---|---|
| **PointNav** | RL (Habitat-Lab + DDPPO) | `habitat-gs` | PointNavResNet-50 + LSTM | `references/task-pointnav.md` |
| **ImageNav** | RL (Habitat-Lab + DDPPO) | `habitat-gs` | PointNavResNet-50 + LSTM | `references/task-imagenav.md` |
| **ObjectNav** | RL (Habitat-Lab + DDPPO) | `habitat-gs` | PointNavResNet-50 + LSTM | `references/task-objectnav.md` |
| **StreamVLN** | VLN (VLM supervised fine-tune) | `habitat-gs-streamvln` | LLaVA-Video-7B-Qwen2 + SigLIP | `references/task-streamvln.md` |
| **Uni-NaVid** | VLN (VLM supervised fine-tune) | `habitat-gs-uni-navid` | Vicuna-7B + EVA-ViT-G | `references/task-uninavid.md` |

**RL tasks (PointNav / ImageNav / ObjectNav)** are the lightweight, fast path — they run in the base `habitat-gs` env and need no external repo. Start here for "quickly train and evaluate a navigation policy."

**VLN tasks (StreamVLN / Uni-NaVid)** are heavy: each needs ≥80 GB VRAM/GPU (StreamVLN has a 24 GB LoRA mode), a separate cloned conda env, and an external repo cloned as a sibling of `habitat-gs/`. Only go here when the user explicitly wants instruction-following VLN.

## The universal flow (all tasks)

1. **Prerequisites** — env + Habitat-Lab + GS data. Read `references/prerequisites.md` and verify before anything else. Skipping this is the #1 cause of failures.
2. **Generate episodes / trajectories** — produces the dataset the policy trains/evals on. The released dataset already ships episodes, so this is OPTIONAL for the standard scenes (the train/eval scripts abort with a clear message if the data is missing).
3. **Train** — `bash scripts_gs/train_<task>.sh --output <dir> [options]`.
4. **Evaluate** — `bash scripts_gs/eval_<task>.sh --ckpt <ckpt> [options]`; read where metrics land in `references/outputs-and-metrics.md`.

## Quick start (PointNav, the simplest task)

```bash
conda activate habitat-gs

# (optional) generate episodes — skip if the released dataset is present
python scripts_gs/generate_pointnav_episodes.py

# train (default 5e8 steps; --output is required)
bash scripts_gs/train_pointnav.sh --output output/pointnav

# evaluate a checkpoint
bash scripts_gs/eval_pointnav.sh --ckpt output/pointnav/checkpoints/ckpt.0.pth
```

ImageNav and ObjectNav are identical in shape — swap `pointnav` for `imagenav` / `objectnav` (their default step counts and episode generators differ; see their references).

## How to use this skill

1. **Confirm the task** with the table above. If the user just says "train a nav policy", default to **PointNav** (fastest to a working result) and say so.
2. **Check prerequisites** (`references/prerequisites.md`): correct conda env, Habitat-Lab installed with the numpy pin patched, GS data under `data/scene_datasets/gs_scenes/`, and for ObjectNav *generation* the SAM + CLIP checkpoints.
3. **Decide whether to generate data.** If `data/scene_datasets/gs_scenes/episodes/<task>/{train,val}/content` already exists, skip generation. Otherwise run the generator (read the task reference — ObjectNav needs extra models; VLN needs a VLM endpoint).
4. **Launch training** with `train_<task>.sh`. Training is long-running — launch it in the background and poll, or hand the user the exact command. For multi-GPU pass `--num-gpus N --num-envs M`. See `references/training-and-finetuning.md` for resume vs. fine-tune vs. encoder-transfer.
5. **Evaluate** with `eval_<task>.sh --ckpt <checkpoint.pth>`. Prefer a single checkpoint **file** — the directory form is a polling watcher that can hang (see `references/task-pointnav.md`). Add `--video-dir DIR` (RL) / `--save-video` (VLN) to record rollouts. For a quick check, append `habitat_baselines.test_episode_count=3`.
6. **Report metrics** by reading them from the eval output (RL: SPL / Success / DistanceToGoal printed to stdout & `eval.stdout`; VLN: SR / SPL / OSR / DTG, Uni-NaVid also writes `summary.json`). See `references/outputs-and-metrics.md`.

## Reference index

| File | Contents |
|---|---|
| `references/prerequisites.md` | Conda envs, Habitat-Lab install + numpy patch, GS data download, ObjectNav SAM/CLIP models |
| `references/data-layout.md` | The shared `gs_scenes/` data root, episode `.json.gz` schema, where every dataset file lives |
| `references/task-pointnav.md` | PointNav generate / train / eval, flags, defaults |
| `references/task-imagenav.md` | ImageNav generate / train / eval, the goal-image quality gate |
| `references/task-objectnav.md` | ObjectNav generate (SAM+CLIP, indoor/outdoor, InteriorGS GT variant) / train / eval |
| `references/task-streamvln.md` | StreamVLN one-time setup, episode + trajectory gen, 3-stage train (+LoRA), eval |
| `references/task-uninavid.md` | Uni-NaVid one-time setup, trajectory gen, stage-1/2 train, eval |
| `references/training-and-finetuning.md` | DDPPO config knobs, resume, fine-tune, encoder transfer, multi-GPU, configs dir |
| `references/outputs-and-metrics.md` | Output run-dir layout, checkpoints, TensorBoard, where eval metrics appear |
| `references/troubleshooting.md` | Common errors and fixes; the cross-task smoke test |

## Smoke test the whole pipeline

`scripts_gs/_verify_full_pipeline.sh` runs train+eval for all five tasks on a tiny 3-train / 3-val subset. It is the authoritative end-to-end command reference, but it **hardcodes a foreign conda path and work dir** — edit those before running. Details in `references/troubleshooting.md`.

## Troubleshooting (quick)

| Problem | Solution |
|---|---|
| `Training episode data not found` | Run the task's `generate_*` script, or download the released Nav Data into `data/scene_datasets/gs_scenes/episodes/` |
| `ModuleNotFoundError: No module named 'habitat'` | Habitat-Lab not installed in this env — see `references/prerequisites.md` |
| `AssertionError: ... number of environments (1) ... mini batches (2)` | Training needs `--num-envs >= 2`; `--num-envs 1` is eval-only. See `references/training-and-finetuning.md` |
| numpy version conflict on `pip install -e habitat-lab` | Patch `habitat-lab/habitat-lab/requirements.txt`: `numpy==1.26.4` → `numpy>=2.0.0,<2.4` |
| GS scene renders blank / CUDA error | habitat-gs must be built with `HABITAT_WITH_CUDA=ON`; GS rendering requires CUDA |
| VLN env / external repo errors | Run the task's `setup_*.sh` first; StreamVLN/Uni-NaVid must be cloned as siblings of `habitat-gs/` |
