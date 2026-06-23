# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: Training episode data not found` (or `Validation episode data not found`) | The task's `episodes/<task>/<split>/content` is missing. Run `python scripts_gs/generate_<task>_episodes.py`, or download the released Nav Data into `data/scene_datasets/gs_scenes/episodes/`. |
| `ModuleNotFoundError: No module named 'habitat'` | Habitat-Lab/habitat-baselines not installed in the active env, or you're loading a `.pth` outside the env. See `prerequisites.md`. |
| `AssertionError: Trainer requires the number of environments (1) to be greater than or equal to the number of trainer mini batches (2)` | Training needs `--num-envs >= 2` (the PPO `num_mini_batch`). `--num-envs 1` is only valid for eval. Either use `--num-envs 2`, or append `habitat_baselines.rl.ppo.num_mini_batch=1`. |
| numpy conflict during `pip install -e habitat-lab` | Patch `habitat-lab/habitat-lab/requirements.txt`: `numpy==1.26.4` → `numpy>=2.0.0,<2.4`, then reinstall. |
| GS scenes render blank / black, or CUDA assertion | habitat-gs must be built with `HABITAT_WITH_CUDA=ON`. GS rendering is CUDA-only; check `python -c "import habitat_sim; print(habitat_sim.cuda_enabled)"`. |
| `ESP_CHECK failed` loading a scene | The scene's `.gs.ply` / `.navmesh` is missing or malformed. Per-scene minimum is `<scene>.gs.ply` + `<scene>.navmesh`. |
| ObjectNav generation: SAM/CLIP not found | Download the checkpoints to `~/.cache/sam_checkpoints/` and `~/.cache/clip_models/` (see `prerequisites.md`). Only needed for ObjectNav *generation*. |
| ObjectNav eval logs a missing `<scene>.gs.scn` | Harmless — semantic descriptor is optional (verified non-fatal). |
| Eval `--ckpt <dir>` hangs, or `RuntimeError: Parent directory data/checkpoints does not exist` | Directory eval is a polling watcher (needs an existing `checkpoint_folder`, and waits for `num_checkpoints` `ckpt.N.pth` files). Evaluate single files instead: `for c in <dir>/ckpt.*.pth; do bash scripts_gs/eval_<task>.sh --ckpt "$c"; done`. See `training-and-finetuning.md` / `task-pointnav.md`. |
| VLN: `setup_*.sh` errors that the repo is missing | StreamVLN / Uni-NaVid must be cloned as siblings of `habitat-gs/` (`../StreamVLN`, `../Uni-NaVid`). |
| VLN: transformers / KV-cache / flash-attn crash | Re-run `setup_vln.sh` / `setup_uninavid.sh` so the compat patch is applied; flash-attn is optional and falls back to eager. |
| VLN OOM | Full fine-tune needs ≥80 GB/GPU. StreamVLN: use `--lora` (24 GB). Reduce `--batch-size` / `--num-frames`. |
| `generate_vln_episodes.py` produces no instructions | It needs a VLM endpoint: set `OPENAI_BASE_URL` + `OPENAI_API_KEY` or pass `--api-config`; or use `--text-only` to skip the VLM. |

## Cross-task smoke test

`scripts_gs/_verify_full_pipeline.sh` runs train+eval for ALL FIVE tasks on a 3-train / 3-val subset — the authoritative end-to-end reference. **It hardcodes a foreign conda install and work dir** (`source /home/yuanhong/miniconda3/...`, `WORK=/data3/...`); edit those to your environment before running. Use it to read off the exact command sequence each task uses, even if you don't run it as-is.
