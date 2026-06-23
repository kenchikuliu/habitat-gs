# Prerequisites

Verify these **before** generating data, training, or evaluating. Missing prerequisites are the most common cause of failures.

## 1. Conda environments

| Env | For | How to create |
|---|---|---|
| `habitat-gs` | RL tasks (PointNav/ImageNav/ObjectNav) + all asset tools | manual (below) |
| `habitat-gs-streamvln` | StreamVLN | created by `bash scripts_gs/setup_vln.sh` (clones `habitat-gs`) |
| `habitat-gs-uni-navid` | Uni-NaVid | created by `bash scripts_gs/setup_uninavid.sh` (clones `habitat-gs`) |

Base env:

```bash
conda create -n habitat-gs python=3.12 cmake=3.27
conda activate habitat-gs
# Install CUDA-compatible torch FIRST (CUDA 12.1 build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

habitat-gs itself must be built with CUDA on (GS rendering requires it):

```bash
HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF pip install .
```

## 2. Habitat-Lab (required for the RL tasks)

PointNav/ImageNav/ObjectNav use Habitat-Lab + habitat-baselines (DDPPO). Install into the **same** `habitat-gs` env.

```bash
git clone https://github.com/facebookresearch/habitat-lab.git
# ⚠️ BEFORE installing, patch the numpy pin so it doesn't conflict with habitat-gs:
#    edit habitat-lab/habitat-lab/requirements.txt
#    numpy==1.26.4   ->   numpy>=2.0.0,<2.4
pip install -e habitat-lab
pip install -e habitat-baselines
```

Symptom if missing: `ModuleNotFoundError: No module named 'habitat'` (also when unpickling `.pth` checkpoints outside the env).

## 3. GS scene data

All tasks read from `data/scene_datasets/gs_scenes/` (HF dataset `RukawaY/gs_scenes`). Per scene only `<scene>.gs.ply` + `<scene>.navmesh` are strictly required. The released dataset also ships ready-made episodes (≈1000/train scene, 100/val scene), so you can skip episode generation for the standard scenes. See `data-layout.md`.

## 4. ObjectNav-only: SAM + CLIP checkpoints (for episode GENERATION only)

Needed only when you run `generate_objectnav_episodes.py` (not for training or eval). Download to:

| Model | Path |
|---|---|
| SAM ViT-B | `~/.cache/sam_checkpoints/sam_vit_b_01ec64.pth` |
| CLIP ViT-B-32 | `~/.cache/clip_models/vit_b_32_laion400m.pt` |

Plus python deps `segment_anything`, `open_clip`, `scikit-learn`, `Pillow`.

## 5. VLN-only: external repos + base models

- **StreamVLN**: clone `https://github.com/InternRobotics/StreamVLN.git` as a **sibling** of `habitat-gs/` (i.e. `../StreamVLN`). `setup_vln.sh` then patches it, installs deps, and downloads LLaVA-Video-7B-Qwen2 + SigLIP into `../StreamVLN/checkpoints/`.
- **Uni-NaVid**: clone `https://github.com/jzhzhang/Uni-NaVid.git` as `../Uni-NaVid`. `setup_uninavid.sh` patches it and downloads EVA-ViT-G + Vicuna-7B-v1.5 + the pretrained Uni-NaVid into `../Uni-NaVid/model_zoo/`.

Both setups support `--proxy URL` for downloads behind a proxy.
