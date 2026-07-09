# Write Your Own Example

This tutorial builds a complete example from scratch: the 1D **heat equation**

$$
\frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial x^2},
\qquad (t, x) \in [0, 1] \times [0, 2\pi],
$$

with periodic boundaries, $\kappa = 0.05$, and $u(0, x) = \sin x$ (exact solution
$e^{-\kappa t} \sin x$). Copying the closest existing example is always a good start — for a
time-dependent scalar PDE that is `examples/allen_cahn`.

## 1. Directory layout

```
examples/heat/
  main.py          # entry point: data, samplers, one call into the trainer
  models.py        # the PDE
  evaluators.py    # problem-specific metrics
  configs/
    base.py        # full hyperparameter tree
    baseline.py    # thin variant
```

## 2. The model (`models.py`)

```python
from functools import partial

import jax.numpy as jnp
from jax import grad, jit, vmap

from jaxpi.models import ForwardIVP


class Heat1D(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, u0, t_star, x_star):
        super().__init__(config, lr, tx, arch, state)
        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star
        self.kappa = 0.05

    def neural_net(self, params, t, x):
        t = t / self.t_star[-1]          # normalize time into [0, 1]
        z = jnp.stack([t, x])
        return self.state.apply_fn(params, z)[0]

    def r_net(self, params, t, x):
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_xx = grad(grad(self.neural_net, argnums=2), argnums=2)(params, t, x)
        return u_t - self.kappa * u_xx   # single component: plain return is fine

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        u_pred = vmap(self.neural_net, (None, None, 0))(params, 0.0, self.x_star)
        ics_loss = jnp.mean((u_pred - self.u0) ** 2)
        res_losses = self.compute_residual_losses(
            params, state, batch,
            pseudo_time=self.config.pseudo_time.enabled,
            causal=self.config.causal.enabled,
        )
        return {"ics": ics_loss, **res_losses}

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_ref):
        u_pred = vmap(vmap(self.neural_net, (None, None, 0)), (None, 0, None))(
            params, self.t_star, self.x_star
        )
        return jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
```

For a **system** of PDEs, return named residuals — see
[the dict-residual convention](/guide/concepts#the-dict-residual-convention).
For a **steady** problem, subclass `ForwardBVP` instead.

## 3. The evaluator (`evaluators.py`)

```python
from jaxpi.evaluator import BaseEvaluator


class Heat1DEvaluator(BaseEvaluator):
    def __call__(self, model, state, loss_dict, batch, u_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)
        if self.config.logging.log_errors:
            self.log_dict["error/l2"] = model.compute_l2_error(state.params, u_ref)
        return self.log_dict
```

Everything logged here lands in W&B and the console table.

## 4. The config (`configs/base.py`)

Copy `examples/allen_cahn/configs/base.py` and adjust; the essentials:

```python
config.input_dim = 2                                 # (t, x)
arch.arch_name = "PirateNet"
arch.fourier_emb = {"embed_scale": 1.0, "embed_dim": 256}
arch.periodicity = {"period": (1.0,), "axis": (1,), "trainable": (False,)}

loss_weighting.loss_weights = {"ics": 1.0, "res": 1.0}   # must match losses()
pseudo_time.pts_weights = {"res": 1.0}                   # must match r_net components
```

## 5. The entry point (`main.py`)

```python
def train_and_evaluate(config):
    u_ref, t_star, x_star = get_dataset()            # or an analytic solution
    dom = jnp.array([[0.0, 1.0], [0.0, 2 * jnp.pi]])

    res_sampler = UniformSampler(dom, batch_size=config.training.batch_size)

    model = create_model(config, models.Heat1D, u0=u_ref[0], t_star=t_star, x_star=x_star)
    evaluator = Heat1DEvaluator(config)

    train(config, model, res_sampler, evaluator=evaluator, eval_args=(u_ref,))
```

Wrap it with the standard absl flags block (copy from any example's `main.py`), then:

```bash
cd examples/heat
python3 main.py --config=configs/baseline.py
```

## 6. Scaling up

- Solution drifts from the IC? Enable `causal.enabled = True`.
- Converging to a trivial steady state? Enable pseudo-time stepping.
- Long or chaotic time horizon? Switch to `train_time_windows`
  (copy the pattern from `examples/ks/main.py`).
- All the knobs are described in [Training Techniques](/guide/training-techniques).
