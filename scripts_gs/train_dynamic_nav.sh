#!/usr/bin/env bash
#
# One-click DDPPO dynamic-navigation training on Gaussian-Splatting scenes.
# Trains an agent to either TRACK (follow) or AVOID a walking GS human avatar.
#
# Usage:
#   bash scripts_gs/train_dynamic_nav.sh --task track --output /path/to/output
#   bash scripts_gs/train_dynamic_nav.sh --task avoid --output /path/to/output --num-envs 8 --num-gpus 2
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
TASK="track"
OUTPUT_DIR=""
NUM_ENVS=4
NUM_GPUS=1
TOTAL_STEPS="5e7"
NUM_CHECKPOINTS=50
PRETRAINED_CKPT=""
EXTRA_ARGS=()

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)            TASK="$2";              shift 2;;
        --output)          OUTPUT_DIR="$2";        shift 2;;
        --num-envs)        NUM_ENVS="$2";          shift 2;;
        --num-gpus)        NUM_GPUS="$2";          shift 2;;
        --total-steps)     TOTAL_STEPS="$2";       shift 2;;
        --num-ckpts)       NUM_CHECKPOINTS="$2";   shift 2;;
        --pretrained-ckpt) PRETRAINED_CKPT="$2";   shift 2;;
        *)                 EXTRA_ARGS+=("$1");     shift;;
    esac
done

case "$TASK" in
    track|avoid|avoid_objectnav|avoid_imagenav) ;;
    *) echo "ERROR: --task must be one of: track avoid avoid_objectnav avoid_imagenav (got '$TASK')"; exit 1;;
esac
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash scripts_gs/train_dynamic_nav.sh --task {track|avoid} --output /path/to/output [options]"
    echo ""
    echo "Options:"
    echo "  --task TASK              track | avoid | avoid_objectnav | avoid_imagenav (default: track)"
    echo "  --output DIR             Output directory for checkpoints and tensorboard (required)"
    echo "  --num-envs N             Parallel environments per GPU (default: 4)"
    echo "  --num-gpus N             GPUs for DDPPO (default: 1)"
    echo "  --total-steps N          Total training steps (default: 5e7)"
    echo "  --num-ckpts N            Number of checkpoints to save (default: 50)"
    echo "  --pretrained-ckpt PATH   Fine-tune from an existing .pth checkpoint"
    echo ""
    echo "Extra arguments are forwarded as Hydra overrides."
    exit 1
fi

CONFIG_NAME="ddppo_dynamic_${TASK}_gs_train"

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Verify data ──────────────────────────────────────────────────────
EP_DIR="data/scene_datasets/gs_scenes/dynamic_nav/episodes"
[[ "$TASK" == "avoid_objectnav" ]] && EP_DIR="data/scene_datasets/gs_scenes/dynamic_nav/episodes_objectnav"
if [[ ! -d "$EP_DIR/train/content" ]]; then
    echo "ERROR: dynamic-nav training episodes not found under $EP_DIR."
    echo "Run:  python scripts_gs/generate_dynamic_nav.py --scenes <scene...> --avatars <id...>   (and generate_dynamic_objectnav.py --scenes <scene...> for objectnav)"
    exit 1
fi

if [[ -n "$PRETRAINED_CKPT" ]]; then
    PRETRAINED_CKPT="$(realpath "$PRETRAINED_CKPT")"
    if [[ ! -f "$PRETRAINED_CKPT" ]]; then
        echo "ERROR: --pretrained-ckpt file not found: $PRETRAINED_CKPT"; exit 1
    fi
    PRETRAINED_CKPT_ADAPTED="$(mktemp -t habitat_ft_ckpt_XXXXXX.pth)"
    trap 'rm -f "$PRETRAINED_CKPT_ADAPTED"' EXIT
    python scripts_gs/_adapt_pretrained_ckpt.py "$PRETRAINED_CKPT" "$PRETRAINED_CKPT_ADAPTED"
    PRETRAINED_CKPT="$PRETRAINED_CKPT_ADAPTED"
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo " DDPPO Dynamic-Nav Training (GS Scenes)"
echo "=========================================="
echo "  Task         : $TASK"
echo "  Config       : $CONFIG_NAME"
echo "  Project root : $PROJECT_ROOT"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Num GPUs     : $NUM_GPUS"
echo "  Num envs/GPU : $NUM_ENVS"
echo "  Total steps  : $TOTAL_STEPS"
echo "  Checkpoints  : $NUM_CHECKPOINTS"
[[ -n "$PRETRAINED_CKPT" ]] && echo "  Pretrained   : $PRETRAINED_CKPT"
echo "=========================================="
echo ""

# ── Hydra overrides ──────────────────────────────────────────────────
OVERRIDES=(
    "--config-name=$CONFIG_NAME"
    "habitat_baselines.num_environments=$NUM_ENVS"
    "habitat_baselines.checkpoint_folder=$OUTPUT_DIR/checkpoints"
    "habitat_baselines.tensorboard_dir=$OUTPUT_DIR/tb"
    "habitat_baselines.log_file=$OUTPUT_DIR/train.log"
    "habitat_baselines.total_num_steps=$TOTAL_STEPS"
    "habitat_baselines.num_checkpoints=$NUM_CHECKPOINTS"
)
if [[ -n "$PRETRAINED_CKPT" ]]; then
    OVERRIDES+=(
        "habitat_baselines.rl.ddppo.pretrained=True"
        "habitat_baselines.rl.ddppo.pretrained_weights=$PRETRAINED_CKPT"
    )
fi

if [[ $NUM_GPUS -gt 1 ]]; then
    python -m torch.distributed.launch --use_env --nproc_per_node "$NUM_GPUS" \
        scripts_gs/run_dynamic_nav.py "${OVERRIDES[@]}" "${EXTRA_ARGS[@]}"
else
    python -u scripts_gs/run_dynamic_nav.py "${OVERRIDES[@]}" "${EXTRA_ARGS[@]}"
fi
