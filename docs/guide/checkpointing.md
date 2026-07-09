# Checkpointing & Resume

JAXPI checkpoints with [Orbax](https://github.com/google/orbax) and makes every example
resumable.

## Layout

Checkpoints live under `<cwd>/<wandb.name>/ckpt` (override the root with
`config.saving.ckpt_path`):

```
baseline/ckpt/                      # single-run examples
  10000/  20000/ ...
baseline/ckpt/time_window_3/        # windowed examples: one manager per window
  10000/ ...
baseline/ckpt/time_window_2_stage_1/  # multi-stage: per window and stage
```

Cadence and retention:

```python
saving.save_every_steps = 10000   # 0 disables checkpointing
saving.num_keep_ckpts = 2
```

A final checkpoint is always written when training ends (including early stops).

## Resume

```bash
python3 main.py --config=configs/baseline.py --config.training.resume=True
```

- **Single-run**: restores the latest step from this run's manager and continues to
  `max_steps`.
- **Time windows**: `latest_time_window` scans the checkpoint directory, restores the last
  trained window, re-propagates the initial condition, and continues the cascade. A partially
  trained window resumes mid-window.
- **Multi-stage**: a window counts as done once its *final stage* is checkpointed; earlier
  finished stages of an interrupted window fast-forward instantly and the interrupted stage
  resumes from its own checkpoint.

## Transfer learning

Windowed training warm-starts each window from the previous window's parameters with a fresh
optimizer state:

```python
training.transfer_learning = True   # False = re-initialize every window
```

## Schedule-free evaluation

With `optim.schedule_free = True`, the parameters in the train state are the *training
iterates*; predictions (error logging, IC propagation, notebooks) should use the averaged
parameters:

```python
from jaxpi.utils import get_eval_params

params = get_eval_params(model.state, config.optim.schedule_free)
```

## Restoring in notebooks

```python
from jaxpi.checkpointing import create_checkpoint_manager, get_ckpt_path, restore_checkpoint

ckpt_mngr = create_checkpoint_manager(config.saving, get_ckpt_path(config))
model.state = restore_checkpoint(ckpt_mngr, model.state)
```

Restored arrays are automatically replicated across all visible devices, so a restored state
can go straight back into (multi-GPU) training.
