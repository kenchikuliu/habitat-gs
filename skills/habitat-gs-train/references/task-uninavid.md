# Uni-NaVid (Vision-and-Language Navigation)

Instruction-following VLN (VLN-R2R), video-based VLA. Backbone Vicuna-7B + EVA-ViT-G. Supervised fine-tune on trajectory videos. Env: `habitat-gs-uni-navid`. Heavy: ≥80 GB VRAM/GPU (A100 80GB recommended). External repo `jzhzhang/Uni-NaVid` cloned as `../Uni-NaVid`.

## Step 1 — One-time setup

```bash
bash scripts_gs/setup_uninavid.sh
```
Clones `habitat-gs` env → `habitat-gs-uni-navid`, installs deps (`transformers>=4.38,<4.46`, peft, deepspeed, decord, flash-attn[optional], …), `pip install -e ../Uni-NaVid --no-deps`, applies `scripts_gs/uninavid_compat.patch`, and downloads EVA-ViT-G + Vicuna-7B-v1.5 + the pretrained Uni-NaVid into `../Uni-NaVid/model_zoo/`. Flags: `--skip-env --skip-deps --skip-patch --skip-download --proxy URL`.

## Step 2 — Generate trajectory data

```bash
conda activate habitat-gs-uni-navid
python scripts_gs/generate_uninavid_trajectories.py --split train --resume
```
Replays each VLN episode (shared `episodes/vln/`) with a greedy follower in a GS-rendering sim, encoding `.mp4` trajectory videos (640×480 @ 10 fps, 120° HFOV) plus LLaVA-conversation annotations. Flags: `--split {train,val,all}` (default all), `--output DIR`, `--gpu N`, `--resume`, `--scenes "a,b"`. Writes `trajectory_data/uninavid/{nav_gs_<split>.json, nav_videos/}`.

## Step 3 — Train (two stages)

```bash
# Recommended: fine-tune from the pretrained Uni-NaVid (default stage)
bash scripts_gs/train_uninavid.sh --output output/uninavid_gs --stage stage-2
# Or train from scratch from Vicuna-7B
bash scripts_gs/train_uninavid.sh --output output/uninavid_gs --stage stage-1
```
`stage-1` auto-selects base `model_zoo/vicuna-7b-v1.5`; `stage-2` (default) auto-selects `model_zoo/uninavid-7b-full-224-video-fps-1-grid-2`. Flags: `--output`(req), `--stage {stage-1,stage-2}`(stage-2), `--num-gpus`(1), `--ckpt`(override base), `--epochs`(1), `--batch-size`(8), `--grad-accum`(2), `--lr`(1e-5); extra args pass through. Needs `nav_gs_train.json` + `eva_vit_g.pth`. Uses DeepSpeed ZeRO-2; runs `../Uni-NaVid/uninavid/train/train_mem.py`.

## Step 4 — Evaluate

```bash
bash scripts_gs/eval_uninavid.sh --ckpt output/uninavid_gs/<checkpoint>
bash scripts_gs/eval_uninavid.sh --ckpt <ckpt> --split val --save-video
```
Flags: `--ckpt`(req), `--output`(default `results/uninavid/<ckpt>_<split>`), `--num-gpus`(1), `--split {train,val}`(val), `--save-video`. Runs the in-repo `scripts_gs/eval_uninavid_gs.py` — a NaVid-VLN-CE-style evaluator that drives the policy online in a `habitat.Env` with `configs/vln_uninavid_gs_eval.yaml`, with rotation/step early-stop. Multi-GPU runs one worker per GPU then an `--aggregate-only` pass. Reports SR, SPL, OSR, DTG (and PL); writes `summary.json` + per-episode `log/stats_*.json`.

## Compat patch

`uninavid_compat.patch` fixes Uni-NaVid for newer transformers (KV-cache `DynamicCache` handling in `uninavid_arch.py`; drops redundant `load_in_4bit` in `builder.py`). Applied by `setup_uninavid.sh`.
