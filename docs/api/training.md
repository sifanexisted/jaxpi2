# jaxpi.training

Shared training loops.

The inner step loop is identical for boundary-value and time-dependent
problems (that distinction lives in ForwardBVP / ForwardIVP), so there is a
single `train_loop`. Time-dependent problems trained forward in time over
consecutive windows use the `train_time_windows` orchestrator on top of it.

## `sample_batches()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/training.py#L30' target='_blank'>[source]</a>

```python
sample_batches(samplers)
```

Infinite batch iterator over a sampler or a dict of samplers.

For a dict, each yielded batch is a dict with one entry drawn from every
sampler (e.g. {"ics": ..., "res": ...}). Any iterable works as a sampler,
so fixed batches can be supplied with itertools.repeat(batch).

## `train_loop()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/training.py#L86' target='_blank'>[source]</a>

```python
train_loop(
    config,
    model,
    batches,
    evaluator=None,
    logger=None,
    ckpt_mngr=None,
    eval_args=(),
    step_offset=0,
    stop_fn=None,
    static_log=None,
)
```

Run up to `config.training.max_steps` optimization steps on `model`.

Args:
  batches: iterator of training batches (see `sample_batches`), or a
    sampler / dict of samplers, which is wrapped automatically.
  evaluator: metrics evaluator; defaults to `BaseEvaluator(config)`.
  logger: console logger; created automatically when omitted.
  ckpt_mngr: optional Orbax checkpoint manager; checkpoints are written
    every `config.saving.save_every_steps` steps and at the end. When
    `config.training.resume` is set and the manager already holds a
    checkpoint, training restores it and continues from its step.
  eval_args: extra positional arguments passed to the evaluator (e.g. the
    reference solution).
  step_offset: added to the step index when logging to W&B, so that
    consecutive windows/stages produce one continuous curve.
  stop_fn: optional `(step, log_dict) -> bool` early-stopping predicate,
    evaluated whenever metrics are logged.
  static_log: optional dict merged into every logged metrics dict (e.g.
    {"time_window": 2}), so run phases are identifiable in W&B charts.

Returns the model (its `state` is updated in place).

## `train()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/training.py#L194' target='_blank'>[source]</a>

```python
train(config, model, batches, evaluator=None, eval_args=(), stop_fn=None)
```

Single-run training: `train_loop` with the standard checkpoint layout.

## `train_time_windows()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/training.py#L210' target='_blank'>[source]</a>

```python
train_time_windows(
    config,
    model,
    make_samplers,
    evaluator=None,
    propagate_ic=None,
    make_eval_args=None,
)
```

Train a time-dependent problem forward over consecutive time windows.

Args:
  make_samplers: `window_idx (0-based) -> sampler or dict of samplers` for
    that window.
  evaluator: metrics evaluator; defaults to `BaseEvaluator(config)`.
  propagate_ic: optional `(model, window_idx)`, invoked after each trained
    window (and after a resume restore) to turn the predicted solution at
    the end of the window into the next window's initial condition —
    typically by updating the arrays that `make_samplers` closes over. It
    may return a replacement model (e.g. rebuilt with the new IC when the
    IC is captured inside jitted closures); returning None keeps the
    current model.
  make_eval_args: optional `window_idx -> tuple` of extra positional
    arguments for the evaluator (e.g. the window's reference solution).

Each window `idx` checkpoints under `time_window_{idx + 1}`.
`config.training.num_time_windows` is the *total* number of windows. When
`config.training.resume` is set, training continues where the checkpoints
end: a partially trained window is re-entered and resumes from its last
step, a finished window's successor starts fresh. When
`config.training.transfer_learning` is set, each window starts from the
previous window's parameters (with a fresh optimizer state); otherwise
from a fresh initialization.
