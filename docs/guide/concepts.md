# Core Concepts

A JAXPI problem is defined by three pieces: a **model** (the PDE), **samplers** (where the
losses are evaluated), and a **config** (every hyperparameter). The shared **trainer** ties
them together.

```
config ──► create_model ──► PINN subclass (your physics)
                │
samplers ───────┴──► train / train_time_windows ──► checkpoints + W&B
```

## Models: `ForwardIVP` and `ForwardBVP`

Your PDE lives in a small subclass that implements three methods:

```python
from jaxpi.models import ForwardIVP

class Burgers(ForwardIVP):
    def neural_net(self, params, t, x):
        """Network output at a single point (scaling, hard constraints, ...)."""
        z = jnp.stack([t, x])
        return self.state.apply_fn(params, z)[0]

    def r_net(self, params, t, x):
        """Pointwise PDE residual (autodiff for derivatives)."""
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_x = grad(self.neural_net, argnums=2)(params, t, x)
        u_xx = grad(grad(self.neural_net, argnums=2), argnums=2)(params, t, x)
        return u_t + u * u_x - 0.01 / jnp.pi * u_xx

    def losses(self, params, state, batch):
        """All loss terms, as a dict."""
        ics_loss = ...  # supervised initial/boundary conditions
        res_losses = self.compute_residual_losses(
            params, state, batch,
            pseudo_time=self.config.pseudo_time.enabled,
            causal=self.config.causal.enabled,
        )
        return {"ics": ics_loss, **res_losses}
```

- Time-dependent problems subclass `ForwardIVP` (adds causal weighting); steady
  boundary-value problems subclass `ForwardBVP`.
- The base class provides the sharded training step, adaptive weight updates, and vmapped
  prediction functions (`sol_pred_fn`, `r_pred_fn`). You never write a training loop.

## The variable-keyed residual convention

Systems of PDEs share **one namespace: the solution variable names**. The model declares
its variables, and each residual is keyed by the variable it evolves in pseudo-time
(momentum-x residual → `u`, continuity → `p`):

```python
class NavierStokes2D(ForwardBVP):
    variables = ("u", "v", "p")   # names of neural_net's outputs, in order

    def r_net(self, params, t, x, y):
        ...
        return {"u": ru, "v": rv, "p": rc}
```

::: warning Why one namespace?
Residuals, pseudo-time weights, and solution components then pair **by key,
automatically** — no ordering conventions (JAX sorts dict keys when flattening pytrees, so
positional pairings silently permute). Unnamed multi-component tuples are rejected with a
clear error.
:::

- `config.pseudo_time.pts_weights` uses the same variable keys: `{"u": 1.0, "v": 1.0, "p": 1.0}`.
- Residual **losses** get a `_res` suffix so they are self-identifying next to other terms:
  loss names follow the uniform `<variable>_<term>` pattern (`u_ic`, `u_bc`, `u_res`), and
  `config.loss_weighting.loss_weights` uses those names.
- Single-component problems skip all of this: return a bare residual and use the `"res"` key.

## Samplers

Samplers are infinite iterators over collocation batches:

| Sampler | Use |
| --- | --- |
| `UniformSampler(dom, batch_size)` | Random points in a box; time-sorted for causal training |
| `MeshSampler(mesh, labels, batch_size)` | Random rows of a point cloud (ICs, BCs, non-rectangular geometry) |
| `TemporalMeshSampler(t_dom, mesh, batch_size)` | Random time × fixed spatial mesh |

Batches can be a single array or a dict of arrays (e.g. `{"ics": ..., "res": ...}`), one entry
per sampler. Fixed batches are just `itertools.repeat(batch)`.

## Losses and adaptive weights

The total loss is $\sum_i \lambda_i \mathcal{L}_i$ over the dict returned by `losses()`. With
`loss_weighting.strategy = "dynamic"`, the $\lambda_i$ are periodically rebalanced so every
term has the same gradient norm — no manual tuning. See
[Training Techniques](/guide/training-techniques) for the full toolbox.

## Multi-device execution

`model.step` shards the batch across all visible devices with `jax.shard_map`, computes
per-device gradients, and averages them with `lax.pmean` — mathematically identical to
single-device training on the full batch (this equivalence is covered by the test suite).
The train state is replicated; no code changes are needed to scale.
