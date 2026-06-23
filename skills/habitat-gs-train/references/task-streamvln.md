# StreamVLN (Vision-and-Language Navigation)

Instruction-following VLN (VLN-R2R). Backbone LLaVA-Video-7B-Qwen2 + SigLIP. Trained by **supervised fine-tuning of a VLM on demonstration trajectories** (not RL). Env: `habitat-gs-streamvln`. Heavy: full fine-tune needs ≥80 GB VRAM/GPU; a `--lora` mode fits on a 24 GB RTX 4090. External repo `InternRobotics/StreamVLN` cloned as `../StreamVLN`.

## Step 1 — One-time setup

```bash
bash scripts_gs/setup_vln.sh
```
Clones the `habitat-gs` env → `habitat-gs-streamvln`, applies `scripts_gs/streamvln_compat.patch` to `../StreamVLN`, installs pinned deps (`transformers==4.45.1`, `accelerate==0.28.0`, deepspeed, peft, …), and downloads LLaVA-Video-7B-Qwen2 + SigLIP into `../StreamVLN/checkpoints/`. Flags: `--skip-env --skip-patch --skip-download --skip-deps --hf-token TOKEN --proxy URL`.

## Step 2 — Generate episodes + trajectories

```bash
conda activate habitat-gs-streamvln
# VLN episodes (R2RVLN-v1): samples navmesh paths, renders waypoint views with GS,
# queries a VLM to write the instruction text. Default 200 train / 50 val per scene.
python scripts_gs/generate_vln_episodes.py
# StreamVLN trajectories: replays each episode with a greedy follower, records (RGB, action).
python scripts_gs/generate_vln_trajectories.py
```
`generate_vln_episodes.py` needs a VLM endpoint via `OPENAI_BASE_URL` + `OPENAI_API_KEY` (or `--api-config config.json`). Useful flags: `--text-only`, `--resume`, `--scenes a,b`, `--train-episodes N`, `--val-episodes N`, `--model`. VLN episodes are SHARED with Uni-NaVid (`episodes/vln/`). Trajectories write to `trajectory_data/vln/` (`annotations.json` + `images/.../rgb/*.jpg`).

## Step 3 — Train (three stages)

```bash
# Standard full fine-tune (≥80 GB)
bash scripts_gs/train_vln.sh --output output/vln_stage1 --stage stage-one
bash scripts_gs/train_vln.sh --output output/vln_dagger --stage dagger   --ckpt output/vln_stage1/checkpoint-XXX
bash scripts_gs/train_vln.sh --output output/vln_stage2 --stage stage-two --ckpt output/vln_dagger/checkpoint-XXX

# LoRA (24 GB)
bash scripts_gs/train_vln.sh --output output/vln_stage1 --stage stage-one --lora
bash scripts_gs/train_vln.sh --output output/vln_dagger --stage dagger --ckpt output/vln_stage1 --lora
```
Stages: `stage-one` = SFT (with data augmentation); `dagger` = retrain with DAgger-collected data (auto-mixes `trajectory_data/vln_dagger/` if present); `stage-two` = co-train with auxiliary QA data. Flags: `--output`(req), `--stage {stage-one|dagger|stage-two}`, `--num-gpus`(1), `--ckpt`, `--epochs`(1), `--batch-size`(2), `--grad-accum`(2), `--lr`(2e-5), `--num-frames`(32), `--lora`. `dagger`/`stage-two` require `--ckpt`. Chained LoRA stages auto-merge the adapter before continuing. Needs `trajectory_data/vln/annotations.json`.

## Step 4 — Evaluate

```bash
bash scripts_gs/eval_vln.sh --ckpt output/vln_stage1/checkpoint-XXX
bash scripts_gs/eval_vln.sh --ckpt <ckpt> --save-video
```
Flags: `--ckpt`(req), `--output`(default `results/vln/<ckpt>_<split>`), `--num-gpus`(1), `--split {train|val}`(val), `--num-frames`(32), `--save-video`. Auto-merges a LoRA checkpoint. Runs `../StreamVLN/streamvln/streamvln_eval.py` with `configs/vln_gs_eval.yaml` (RGB+Depth 640×480, forward 0.25 m, turn 15°, success 3.0 m). Metrics: Success, SPL, Oracle Success, Distance-to-Goal, Oracle Navigation Error.

## Compat patch

`streamvln_compat.patch` fixes StreamVLN for habitat-lab 0.3.3 (removes `try_cv2_import`, honors `--vision_tower`, fixes SigLIP delay-load + flash-attn fallback). Applied by `setup_vln.sh`.
