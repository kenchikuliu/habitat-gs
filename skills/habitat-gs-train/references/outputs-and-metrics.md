# Outputs and metrics

## RL task output run dir (`--output`)

```
output/<task>/
├── checkpoints/    # ckpt.0.pth, ckpt.1.pth, ..., latest.pth, .resume_state.pth
├── tb/             # TensorBoard event files
└── train.log       # training log
```
Checkpoints are torch pickles that require the `habitat` module to unpickle (load them inside the `habitat-gs` env). `--num-ckpts` controls how many are written across the run.

Hydra also writes a per-invocation snapshot under `outputs/<YYYY-MM-DD>/<HH-MM-SS>/.hydra/` (`config.yaml` = fully-resolved config, `overrides.yaml`, `hydra.yaml`) plus `run_<task>.log` — useful to see exactly what config a run used.

Watch training live:
```bash
tensorboard --logdir output/<task>/tb
```

## RL eval metrics — stdout / log only (no JSON)

The RL evaluators print averaged metrics to stdout (and the run's `eval.stdout` if you tee it). There is **no metrics JSON file** for PointNav/ImageNav/ObjectNav. Example:
```
Average episode reward: -0.0617
Average episode spl: 0.0000
Average episode distance_to_goal: 12.7954
Average episode success: 0.0000
Average episode soft_spl: 0.0046
```
To capture metrics, redirect: `bash scripts_gs/eval_pointnav.sh --ckpt <ckpt> | tee eval.stdout`. Rollout videos go to `--video-dir` when set.

## VLN eval outputs

- **StreamVLN**: results written under `--output` (default `results/vln/<ckpt>_<split>`). Metrics Success / SPL / OSR / DTG / ONE.
- **Uni-NaVid**: `scripts_gs/eval_uninavid_gs.py` writes `summary.json` (`num_episodes, SR, OSR, SPL, DTG, PL`) plus per-episode `log/stats_*.json` under `--output` (default `results/uninavid/<ckpt>_<split>`).

## Reference: pre-existing example runs in the repo

`output/verify_{pointnav,imagenav,objectnav}/` are small verification runs (each with `checkpoints/`, `tb/`, `train.log`, `eval.stdout`) — handy as a shape reference for what a finished run looks like.
