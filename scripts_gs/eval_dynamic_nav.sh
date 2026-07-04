#!/usr/bin/env bash
#
# One-click DDPPO dynamic-navigation evaluation on Gaussian-Splatting scenes.
#
# Usage:
#   bash scripts_gs/eval_dynamic_nav.sh --task track --ckpt /path/to/checkpoint.pth
#   bash scripts_gs/eval_dynamic_nav.sh --task avoid --ckpt /path/to/checkpoints_dir --video-dir /path/to/videos
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
TASK="track"
CKPT_PATH=""
NUM_ENVS=1
VIDEO_DIR=""
EXTRA_ARGS=()

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)       TASK="$2";         shift 2;;
        --ckpt)       CKPT_PATH="$2";    shift 2;;
        --num-envs)   NUM_ENVS="$2";     shift 2;;
        --video-dir)  VIDEO_DIR="$2";    shift 2;;
        *)            EXTRA_ARGS+=("$1"); shift;;
    esac
done

case "$TASK" in
    track|avoid|avoid_objectnav|avoid_imagenav) ;;
    *) echo "ERROR: --task must be one of: track avoid avoid_objectnav avoid_imagenav (got '$TASK')"; exit 1;;
esac
if [[ -z "$CKPT_PATH" ]]; then
    echo "Usage: bash scripts_gs/eval_dynamic_nav.sh --task {track|avoid} --ckpt /path/to/checkpoint [options]"
    echo ""
    echo "Options:"
    echo "  --task TASK            track | avoid | avoid_objectnav | avoid_imagenav (default: track)"
    echo "  --ckpt PATH            Path to checkpoint .pth file or directory (required)"
    echo "  --num-envs N           Number of parallel environments (default: 1)"
    echo "  --video-dir DIR        Directory to save evaluation videos (optional)"
    echo ""
    echo "Extra arguments are forwarded as Hydra overrides."
    exit 1
fi

CONFIG_NAME="ddppo_dynamic_${TASK}_gs_eval"

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Verify data ──────────────────────────────────────────────────────
EP_DIR="data/scene_datasets/gs_scenes/dynamic_nav/episodes"
[[ "$TASK" == "avoid_objectnav" ]] && EP_DIR="data/scene_datasets/gs_scenes/dynamic_nav/episodes_objectnav"
if [[ ! -d "$EP_DIR/val/content" ]]; then
    echo "ERROR: dynamic-nav validation episodes not found under $EP_DIR."
    echo "Run:  python scripts_gs/generate_dynamic_nav.py --scenes <scene...> --avatars <id...>   (and generate_dynamic_objectnav.py --scenes <scene...> for objectnav)"
    exit 1
fi

CKPT_PATH="$(realpath "$CKPT_PATH")"
if [[ ! -e "$CKPT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"; exit 1
fi

echo "=========================================="
echo " DDPPO Dynamic-Nav Evaluation (GS Scenes)"
echo "=========================================="
echo "  Task         : $TASK"
echo "  Config       : $CONFIG_NAME"
echo "  Project root : $PROJECT_ROOT"
echo "  Checkpoint   : $CKPT_PATH"
echo "  Num envs     : $NUM_ENVS"
[[ -n "$VIDEO_DIR" ]] && echo "  Video dir    : $VIDEO_DIR"
echo "=========================================="
echo ""

# ── Hydra overrides ──────────────────────────────────────────────────
OVERRIDES=(
    "--config-name=$CONFIG_NAME"
    "habitat_baselines.num_environments=$NUM_ENVS"
    "habitat_baselines.eval_ckpt_path_dir=$CKPT_PATH"
)
if [[ -n "$VIDEO_DIR" ]]; then
    mkdir -p "$VIDEO_DIR"
    OVERRIDES+=(
        "habitat_baselines.video_dir=$VIDEO_DIR"
        "habitat_baselines.eval.video_option=[disk]"
    )
fi

python -u scripts_gs/run_dynamic_nav.py "${OVERRIDES[@]}" "${EXTRA_ARGS[@]}"
