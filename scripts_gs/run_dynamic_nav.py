#!/usr/bin/env python3
"""Hydra entry-point for dynamic-navigation (avoidance / tracking) train & eval on GS scenes.

Registers the GS config directory (``data/scene_datasets/gs_scenes/configs/``) as a Hydra
search path so the ``ddppo_dynamic_*_gs_*`` YAMLs are found directly.

Usage:
    # tracking training (default)
    python scripts_gs/run_dynamic_nav.py
    # avoidance training
    python scripts_gs/run_dynamic_nav.py --config-name=ddppo_dynamic_avoid_gs_train
    # evaluation
    python scripts_gs/run_dynamic_nav.py --config-name=ddppo_dynamic_track_gs_eval \
        habitat_baselines.eval_ckpt_path_dir=/path/to/ckpt.pth
"""
import random
import sys

import numpy as np
import torch

# ── Register Hydra search-path plugins BEFORE @hydra.main ────────────
from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

register_hydra_plugin(HabitatBaselinesConfigPlugin)
register_hydra_plugin(HabitatConfigPlugin)

import hydra
from omegaconf import DictConfig

from habitat.config.default import patch_config
from habitat_baselines.run import execute_exp


@hydra.main(
    version_base=None,
    # Relative to THIS file → ../data/scene_datasets/gs_scenes/configs
    config_path="../data/scene_datasets/gs_scenes/configs",
    config_name="ddppo_dynamic_track_gs_train",
)
def main(cfg: DictConfig) -> None:
    cfg = patch_config(cfg)
    random.seed(cfg.habitat.seed)
    np.random.seed(cfg.habitat.seed)
    torch.manual_seed(cfg.habitat.seed)
    if (
        cfg.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()
