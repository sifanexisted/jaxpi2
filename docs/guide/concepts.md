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

## The dict-residual convention

Systems of PDEs return **named** residuals:

```python
def r_net(self, params, t, x, y):
    ...
    return {"ru": ru, "rv": rv, "rc": rc}
```

::: warning Why a dict?
Loss terms and pseudo-time weights are matched to residual components **by name**. Returning
an unnamed tuple would leave the pairing to dict iteration order (alphabetical for configs),
which silently mislabels losses — JAXPI rejects multi-component tuples with a clear error.
:::

The keys must match `config.pseudo_time.pts_weights` (and appear in
`config.loss_weighting.loss_weights`).

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
