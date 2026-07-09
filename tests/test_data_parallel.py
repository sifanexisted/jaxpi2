"""Sharded-vs-unsharded equivalence with ALL weighting systems active.

The suite's other equivalence tests exercise one mechanism at a time. Here a
single stress model combines everything that interacts with the sharded batch:

- dict batches ({"ics": (coords, values), "res": points}) sharded per leaf,
- two named residual components (non-alphabetical dict keys),
- causal weighting (all_gather across devices in global time order),
- pseudo-time stepping (prev_params shift inside the causal residual),
- dynamic grad-norm loss weights,

and verifies that the 8-fake-device sharded implementation reproduces a plain
single-device reference, including over multi-step trajectories.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, jit, value_and_grad, vmap

from jaxpi.models import ForwardIVP

from helpers import make_batch, make_config, make_model


class StressIVP(ForwardIVP):
    """Two-component IVP with ICs flowing through the (sharded) batch."""

    def neural_net(self, params, t, x):
        z = jnp.stack([t, x])
        out = self.state.apply_fn(params, z)
        return out[0], out[1]

    def u_net(self, params, t, x):
        return self.neural_net(params, t, x)[0]

    def v_net(self, params, t, x):
        return self.neural_net(params, t, x)[1]

    def r_net(self, params, t, x):
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        u_x = grad(self.u_net, argnums=2)(params, t, x)
        v_t = grad(self.v_net, argnums=1)(params, t, x)
        v_x = grad(self.v_net, argnums=2)(params, t, x)
        # deliberately non-alphabetical key order
        return {"rb": u_t + u_x, "ra": v_t - 0.5 * v_x}

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        coords, values = batch["ics"]
        u_pred, v_pred = vmap(self.neural_net, (None, None, 0))(params, 0.0, coords)
        u_ic_loss = jnp.mean((u_pred - values[:, 0]) ** 2)
        v_ic_loss = jnp.mean((v_pred - values[:, 1]) ** 2)

        res_losses = self.compute_residual_losses(
            params,
            state,
            batch["res"],
            pseudo_time=self.config.pseudo_time.enabled,
            causal=self.config.causal.enabled,
        )
        return {"u_ic": u_ic_loss, "v_ic": v_ic_loss, **res_losses}


def stress_config():
    return make_config(
        out_dim=2,
        loss_weights={"u_ic": 1.0, "v_ic": 1.0, "ra": 1.0, "rb": 1.0},
        pts_weights={"ra": 1.0, "rb": 1.0},
        causal=True,
        num_chunks=8,
        pseudo_time=True,
    )


def stress_batch(key, n_res=64, n_ics=32):
    key1, key2, key3 = jax.random.split(key, 3)
    res = make_batch(key1, n=n_res)  # time-sorted (n_res, 2)
    coords = jax.random.uniform(key2, (n_ics,))
    values = jax.random.normal(key3, (n_ics, 2))
    return {"ics": (coords, values), "res": res}


def reference_step(model, state, batch):
    """Mirror of the sharded training step on the full, unsharded batch."""
    prev_params = state.params
    (loss, loss_dict), grads = value_and_grad(model.loss, has_aux=True)(
        state.params, state, batch
    )
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        prev_params=prev_params,
    )
    return state, loss, loss_dict


def assert_trees_close(actual, expected, rtol=1e-4, atol=1e-6):
    for got, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(ref), rtol=rtol, atol=atol
        )


def test_combined_weightings_single_step():
    """One sharded step with causal + pseudo-time + dict batch == reference."""
    model = make_model(stress_config(), model_cls=StressIVP)
    state = model.state
    batch = stress_batch(jax.random.PRNGKey(0))

    ref_state, ref_loss, ref_dict = reference_step(model, state, batch)
    new_state, loss, loss_dict = model.step(state, batch)

    np.testing.assert_allclose(float(loss), float(ref_loss), rtol=1e-5)
    for key in ref_dict:
        np.testing.assert_allclose(
            float(loss_dict[key]), float(ref_dict[key]), rtol=1e-5,
            err_msg=f"loss term {key}",
        )
    assert_trees_close(new_state.params, ref_state.params)


def test_combined_weightings_trajectory():
    """A multi-step trajectory with scheduled weight updates stays identical.

    Exercises prev_params propagation (the pseudo-time shift changes every
    step) and both weight-update functions on the evolved state.
    """
    model = make_model(stress_config(), model_cls=StressIVP)

    sharded_state = model.state
    ref_state = model.state
    init_state = model.state
    key = jax.random.PRNGKey(1)

    for step_idx in range(6):
        key, subkey = jax.random.split(key)
        batch = stress_batch(subkey)

        sharded_state, s_loss, _ = model.step(sharded_state, batch)
        ref_state, r_loss, _ = reference_step(model, ref_state, batch)
        np.testing.assert_allclose(float(s_loss), float(r_loss), rtol=2e-4)

        if step_idx % 2 == 1:
            # pts weights: sharded update vs direct (unsharded) computation
            ref_pts = model.compute_pts_weights(ref_state, init_state, batch["res"])
            expected = {
                k: ref_state.pts_weights[k] * ref_state.pts_momentum
                + (1 - ref_state.pts_momentum) * ref_pts[k]
                for k in ref_pts
            }
            sharded_state = model.update_pts_weights(
                sharded_state, init_state, batch["res"]
            )
            ref_state = ref_state.apply_pts_weights(pts_weights=ref_pts)
            assert_trees_close(sharded_state.pts_weights, expected, rtol=2e-3)
            assert_trees_close(sharded_state.pts_weights, ref_state.pts_weights, rtol=2e-3)

            # loss weights: sharded update vs direct computation
            ref_lw = model.compute_loss_weights(ref_state, batch)
            sharded_state = model.update_loss_weights(sharded_state, batch)
            ref_state = ref_state.apply_loss_weights(loss_weights=ref_lw)
            assert_trees_close(
                sharded_state.loss_weights, ref_state.loss_weights, rtol=2e-3
            )

    # After 6 steps + 3 rounds of weight updates the parameters still agree
    assert_trees_close(sharded_state.params, ref_state.params, rtol=1e-3, atol=1e-5)
    assert int(sharded_state.step) == int(ref_state.step) == 6


def test_ics_are_actually_sharded():
    """Sanity check on the mechanism: each device must see 1/8 of the IC batch.

    Computes a shard-size-dependent quantity (sum over the local IC shard,
    NOT divided by the global size) inside shard_map and checks it differs
    from the unsharded value — guarding against silent full replication.
    """
    from jax import lax
    from jax.sharding import PartitionSpec as P

    model = make_model(stress_config(), model_cls=StressIVP)
    coords = jnp.arange(32.0)

    @jax.jit
    def local_count(coords):
        def inner(c):
            return jnp.full((1,), c.shape[0])

        return jax.shard_map(
            inner, mesh=model.mesh, in_specs=P("batch"), out_specs=P("batch"),
            check_vma=False,
        )(coords)

    counts = np.asarray(local_count(model._shard_batch(coords)))
    assert counts.tolist() == [4] * 8  # 32 points over 8 devices
