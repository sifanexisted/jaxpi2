# Evaluation

Evaluation happens in two places: **during training** (the evaluator logs metrics to W&B and
the console) and **after training** (restore a checkpoint and analyze the solution, typically
in the example's `eval.ipynb`).

## During training: evaluators

Each example defines an evaluator in `evaluators.py` that extends `BaseEvaluator` with
problem-specific metrics — most commonly a relative $L^2$ error against the reference
solution:

```python
from jaxpi.evaluator import BaseEvaluator

class KSEvaluator(BaseEvaluator):
    def __call__(self, model, state, loss_dict, batch, u_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)
        if self.config.logging.log_errors:
            self.log_dict["l2_error"] = model.compute_l2_error(state.params, u_ref)
        return self.log_dict
```

The base class already logs losses, learning rate, adaptive loss/pseudo-time weights, causal
gates, and (optionally) raw unweighted residuals — all toggled in `config.logging`. Extra
arguments (like `u_ref` above) are passed through the trainer's `eval_args`.

The error itself is usually a jitted method on the model, vmapped over the reference grid:

```python
@partial(jit, static_argnums=(0,))
def compute_l2_error(self, params, u_ref):
    u_pred = vmap(vmap(self.neural_net, (None, None, 0)), (None, 0, None))(
        params, self.t_star, self.x_star
    )
    return jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
```

## After training: restoring a model

Rebuild the model exactly as `main.py` does, then restore the latest checkpoint:

```python
from jaxpi.models import create_model
from jaxpi.checkpointing import create_checkpoint_manager, get_ckpt_path, restore_checkpoint
from configs import baseline

config = baseline.get_config()
model = create_model(config, models.KS, u0=u0, t_star=t_star, x_star=x_star)

ckpt_mngr = create_checkpoint_manager(config.saving, get_ckpt_path(config))
model.state = restore_checkpoint(ckpt_mngr, model.state)   # latest step
# restore_checkpoint(ckpt_mngr, model.state, step=50000)   # or a specific step
```

For time-windowed runs, restore per window with
`create_checkpoint_manager(config.saving, ckpt_path, suffix="time_window_3")`.

::: warning Schedule-free runs
With `optim.schedule_free = True`, `state.params` holds the raw training iterates. Always
predict with the averaged parameters:

```python
from jaxpi.utils import get_eval_params
params = get_eval_params(model.state, config.optim.schedule_free)
```
:::

## Predicting on a grid

Every model exposes vmapped prediction functions built from `neural_net`:

```python
# space-time grid for a 1D problem
u_pred = vmap(vmap(model.neural_net, (None, None, 0)), (None, 0, None))(
    params, t_star, x_star
)

error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
```

Derived quantities come from differentiating the network, exactly like in `r_net` — e.g.
vorticity in the Kolmogorov flow example is `grad`-computed from the velocity outputs rather
than being a network output. For large grids (e.g. $256^3$ points in Taylor–Green), evaluate
in batches — see `predict_in_batches` in `examples/taylor_green/utils.py`.

## The notebooks

Each example ships an `eval.ipynb` that does all of the above and plots predictions against
the reference:

```bash
cd examples/ks
jupyter lab eval.ipynb
```

The typical structure: load the config and dataset → `create_model` → restore the checkpoint
→ compute the relative $L^2$ error → plot predicted vs reference fields and the pointwise
error.

## Related

- [Checkpointing & Resume](/guide/checkpointing) — checkpoint layout and resume semantics
- [Write Your Own Example](/guide/new-example) — includes wiring up an evaluator
