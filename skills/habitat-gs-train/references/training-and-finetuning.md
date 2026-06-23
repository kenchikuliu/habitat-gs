# Training internals, fine-tuning, multi-GPU (RL tasks)

Applies to the three Habitat-Lab DDPPO tasks (PointNav / ImageNav / ObjectNav). Entry point is `scripts_gs/run_<task>.py` (Hydra), driven by the `train_<task>.sh` / `eval_<task>.sh` wrappers.

## Hydra overrides the wrappers set

`train_<task>.sh` builds these automatically from its flags:
```
habitat_baselines.num_environments=<--num-envs>
habitat_baselines.checkpoint_folder=<--output>/checkpoints
habitat_baselines.tensorboard_dir=<--output>/tb
habitat_baselines.log_file=<--output>/train.log
habitat_baselines.total_num_steps=<--total-steps>
habitat_baselines.num_checkpoints=<--num-ckpts>
```
Anything else you append to the command is forwarded verbatim as a Hydra override. `eval_<task>.sh` adds `--config-name=ddppo_<task>_gs_eval`, `habitat_baselines.eval_ckpt_path_dir=<--ckpt>`, and (with `--video-dir`) `habitat_baselines.video_dir` + `eval.video_option=[disk]`.

## Resume an interrupted run

Just re-launch with the **same `--output`** directory. habitat-baselines auto-detects `<output>/checkpoints/.resume_state.pth` and continues — optimizer state, step counter and seeds are preserved.

## Fine-tune from a checkpoint

```bash
bash scripts_gs/train_pointnav.sh \
    --output output/pointnav_ft \
    --pretrained-ckpt output/pointnav/checkpoints/ckpt.99.pth
```
The wrapper runs `scripts_gs/_adapt_pretrained_ckpt.py` to prefix the state_dict keys (so habitat-baselines' `pretrained_weights` loader reconstructs them), then sets `ddppo.pretrained=True` + `ddppo.pretrained_weights=<adapted>`. By default it loads the **whole** policy and **re-initializes the critic**. Optimizer/step/seed are reset — this is fine-tuning, not resume.

Customize via appended Hydra overrides:
```bash
# keep the trained critic
... --pretrained-ckpt <ckpt> habitat_baselines.rl.ddppo.reset_critic=False

# encoder-only transfer (e.g. PointNav -> ImageNav), freeze the encoder
... --pretrained-ckpt <ckpt> \
    habitat_baselines.rl.ddppo.pretrained=False \
    habitat_baselines.rl.ddppo.pretrained_encoder=True \
    habitat_baselines.rl.ddppo.train_encoder=False
```

## Multi-GPU

`--num-gpus N` (N>1) launches `python -m torch.distributed.launch --use_env --nproc_per_node N scripts_gs/run_<task>.py ...`. Scale `--num-envs` per GPU too, e.g. `--num-envs 8 --num-gpus 4`.

## Minimum `--num-envs` and reducing the footprint

For **training**, `--num-envs` must be **>= `ppo.num_mini_batch` (default 2)** — habitat-baselines asserts `num_environments >= num_mini_batch` and crashes otherwise (`--num-envs 1` is rejected; it is only valid for *eval*, where it is the default). Verified: `--num-envs 1` training aborts with `AssertionError: Trainer requires the number of environments (1) to be greater than or equal to the number of trainer mini batches (2)`; `--num-envs 2` runs fine and writes checkpoints.

To shrink GPU memory: keep `--num-envs 2` (the floor), and/or append Hydra overrides such as `habitat_baselines.rl.ppo.num_mini_batch=1` (then `--num-envs 1` is allowed), `habitat_baselines.rl.ppo.num_steps=64`, or lower the sensor resolution. A quick smoke run that completes in well under a minute and writes one checkpoint:
```bash
bash scripts_gs/train_pointnav.sh --output output/_smoke --num-envs 2 --total-steps 256 --num-ckpts 1
```

## Shared config facts

Configs live in `data/scene_datasets/gs_scenes/configs/` (this is the Hydra search path — `run_<task>.py` points `config_path` there directly, no symlinks). Shared defaults: `scenes_dir: data/scene_datasets`, `simulator.default_agent_navmesh: False` (use the shipped `.navmesh`), `trainer_name: ddppo`, backbone `resnet50`, `rnn_type LSTM`, 2 recurrent layers, PPO `lr=2.5e-4 clip=0.2 gamma=0.99 hidden=512`, `force_torch_single_threaded: True`. Eval configs additionally set `evaluate=True`, `num_environments=1`, `eval.use_ckpt_config=False` (eval uses the eval YAML, not the checkpoint's saved config).

There are extra non-standard PointNav configs in `configs/` (`ddppo_pointnav_{50,100}gs.yaml`, `..._100mesh.yaml`, `..._*_ft*.yaml`, `..._20gs_mesh_avatar_ft5e6.yaml`) — these are ablation/scaling experiments and are NOT used by `train_pointnav.sh` (which forces `--config-name=ddppo_pointnav_gs_train`). To run one, call `run_pointnav.py` directly with `--config-name=<that file>`.
