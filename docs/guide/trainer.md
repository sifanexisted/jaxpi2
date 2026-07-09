# The Trainer

All examples share one training implementation in `jaxpi/training.py`. The inner step loop is
identical for boundary-value and time-dependent problems — what differs is orchestration
around it.

## `train` — single-run problems

```python
from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

model = create_model(config, models.Burgers, u0=u0, t_star=t_star, x_star=x_star)
evaluator = BurgersEvaluator(config)

train(config, model, UniformSampler(dom, config.training.batch_size),
      evaluator=evaluator, eval_args=(u_ref,))
```

`create_model` builds the LR schedule, optimizer, architecture, and train state from the
config and forwards extra arguments to your model class. `train` then runs
`config.training.max_steps` steps with:

- scheduled adaptive **loss-weight** and **pseudo-time-weight** updates,
- periodic evaluation → W&B + console logging (`eval_args` are passed through to your evaluator),
- periodic + final **checkpoints** under `<cwd>/<wandb.name>/ckpt`,
- automatic **resume** when `config.training.resume` is set,
- an optional `stop_fn(step, log_dict)` early-stopping predicate.

## `train_time_windows` — forward-in-time curriculum

Chaotic and strongly transient problems are trained window by window: fit $[0, \Delta T]$,
predict the solution at $\Delta T$, use it as the next initial condition, repeat.

```python
from jaxpi.training import train_time_windows

def make_samplers(window_idx):
    ics_sampler = MeshSampler(mesh, ic_values, batch_size)
    res_sampler = UniformSampler(dom, batch_size)
    return {"ics": iter(ics_sampler), "res": iter(res_sampler)}

def propagate_ic(model, window_idx):
    # your prediction of the field at t = ΔT becomes the next window's IC
    ...

train_time_windows(config, model, make_samplers,
                   evaluator=evaluator, propagate_ic=propagate_ic)
```

Per window, the orchestrator creates a checkpoint manager (`time_window_{i}`), optionally
transfers the previous window's parameters into a fresh optimizer state
(`config.training.transfer_learning`), trains with `train_loop`, and calls your
`propagate_ic` hook. With `config.training.resume`, it restores the last trained window and
continues the cascade.

::: tip Rebuilding the model in `propagate_ic`
If your initial condition is stored as a model attribute (rather than flowing through the
batch), return a **new model instance** from `propagate_ic` — jitted step functions bake
constants in at trace time and will not see attribute mutations.
:::

## `train_loop` — the shared inner loop

Both entry points delegate to `train_loop(config, model, batches, ...)`, which you can call
directly for custom orchestration (the multi-stage Taylor–Green driver does exactly this).
`batches` is any iterator: a sampler, a dict of samplers, `itertools.repeat(fixed_batch)`, or
a generator that augments batches on the fly (e.g. rejection-sampled points).

## What a training step does

```
batch ──► shard across devices ──► value_and_grad(loss) ──► pmean ──► optimizer update
                                        │
             every loss_weighting.update_schedule steps: grad-norm rebalancing
             every pseudo_time.update_schedule steps:    pseudo-time weight update
```
