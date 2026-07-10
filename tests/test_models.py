import jax
import numpy as np
import optax
import pytest
from jax import value_and_grad

from jaxpi.models import create_arch, create_lr_schedule, create_optimizer

from helpers import (
    TinyIVP,
    TupleTwoComponentIVP,
    TwoComponentIVP,
    make_batch,
    make_config,
    make_model,
)


def test_runs_on_eight_fake_devices():
    assert jax.device_count() == 8


class TestFactories:
    def test_create_arch_unknown_raises(self):
        cfg = make_config().arch
        cfg.arch_name = "transformer"
        with pytest.raises(NotImplementedError):
            create_arch(cfg)

    def test_create_lr_schedule_unknown_raises(self):
        cfg = make_config().optim
        cfg.lr_schedule = "constant"
        with pytest.raises(NotImplementedError):
            create_lr_schedule(cfg)

    def test_create_optimizer_unknown_raises(self):
        cfg = make_config().optim
        cfg.optimizer = "sgd"
        with pytest.raises(NotImplementedError):
            create_optimizer(cfg, 1e-3)


def test_train_state_fields():
    model = make_model(make_config())
    state = model.state
    assert set(state.loss_weights) == {"ics", "res"}
    assert set(state.pts_weights) == {"res"}
    assert state.momentum == 0.9
    assert state.pts_momentum == 0.9


def test_training_reduces_loss():
    model = make_model(make_config())
    state = model.state
    key = jax.random.PRNGKey(0)

    first_loss = None
    for _ in range(200):
        key, subkey = jax.random.split(key)
        batch = make_batch(subkey)
        state, loss, _ = model.step(state, batch)
        if first_loss is None:
            first_loss = loss

    assert np.isfinite(loss)
    assert loss < first_loss


def test_sharded_step_matches_unsharded_reference():
    """Params/losses from the 8-device sharded step must match a plain
    single-device update on the full batch (regression for missing pmean)."""
    model = make_model(make_config())
    state = model.state
    batch = make_batch(jax.random.PRNGKey(1))

    (ref_loss, ref_dict), grads = value_and_grad(model.loss, has_aux=True)(
        state.params, state, batch
    )
    updates, _ = state.tx.update(grads, state.opt_state, state.params)
    ref_params = optax.apply_updates(state.params, updates)

    new_state, loss, loss_dict = model.step(state, batch)

    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
    for key in ref_dict:
        np.testing.assert_allclose(loss_dict[key], ref_dict[key], rtol=1e-5)
    for ref_leaf, leaf in zip(
        jax.tree.leaves(ref_params), jax.tree.leaves(new_state.params)
    ):
        np.testing.assert_allclose(
            np.asarray(leaf), np.asarray(ref_leaf), rtol=1e-4, atol=1e-6
        )


def test_causal_loss_sharded_matches_unsharded():
    """The all_gather causal path (8 devices) must equal the single-device
    causal loss on the full time-sorted batch."""
    model = make_model(make_config(causal=True, num_chunks=8))
    state = model.state
    batch = make_batch(jax.random.PRNGKey(2), n=128)

    ref_loss, ref_dict = model.loss(state.params, state, batch)
    _, loss, loss_dict = model.step(state, batch)

    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
    for key in ref_dict:
        np.testing.assert_allclose(loss_dict[key], ref_dict[key], rtol=1e-5)


def _causal_reference(chunk_loss, tol):
    """gamma_i = exp(-tol * sum of losses of chunks EARLIER than i)."""
    num_chunks = len(chunk_loss)
    gammas = np.exp(
        -tol * np.array([chunk_loss[:i].sum() for i in range(num_chunks)])
    )
    return float(np.mean(chunk_loss * gammas))


def test_causal_weights_gate_by_earlier_chunks():
    """Regression for the triu.T inversion: chunks must be gated by the
    cumulative loss of EARLIER chunks, not later ones."""
    model = make_model(make_config(causal=True, num_chunks=8))

    # Per-chunk constant residuals with uneven magnitudes; the asymmetry
    # makes correct and time-inverted gating produce different values.
    c = np.array([3.0, 1.0, 0.5, 0.0, 0.0, 0.0, 2.0, 0.1])
    res = np.repeat(c, 4)[None, :]  # (1 component, 32 time-sorted points)

    got = float(model._causal_losses(jax.numpy.asarray(res))[0])

    chunk_loss = c**2
    expected = _causal_reference(chunk_loss, tol=1.0)
    np.testing.assert_allclose(got, expected, rtol=1e-5)

    # Anti-causal gating (the old bug) gives a distinctly different value
    inverted_gammas = np.exp(
        -np.array([chunk_loss[i + 1 :].sum() for i in range(8)])
    )
    inverted = float(np.mean(chunk_loss * inverted_gammas))
    assert not np.isclose(got, inverted, rtol=1e-3)


def test_causal_chunks_must_divide_devices():
    model = make_model(make_config(causal=True, num_chunks=4))  # 8 devices
    batch = make_batch(jax.random.PRNGKey(3))
    with pytest.raises(AssertionError, match="divisible"):
        model.step(model.state, batch)


def test_residual_losses_follow_dict_keys():
    """Regression for alphabetical mislabeling: loss keys must come from the
    r_net dict (whatever its order), not ConfigDict's sorted key order."""
    config = make_config(
        out_dim=2,
        loss_weights={"ra": 1.0, "rb": 1.0},
        pts_weights={"ra": 1.0, "rb": 1.0},
    )
    model = make_model(config, model_cls=TwoComponentIVP)
    batch = make_batch(jax.random.PRNGKey(4))

    loss_dict = model.compute_residual_losses(model.state.params, model.state, batch)

    # r_net returns {"rb": 0, "ra": 3}; a zip against alphabetically sorted
    # pts_weights keys would swap the two losses.
    np.testing.assert_allclose(loss_dict["rb"], 0.0, atol=1e-6)
    np.testing.assert_allclose(loss_dict["ra"], 9.0, rtol=1e-5)


def test_multi_component_tuple_return_raises():
    """Unnamed multi-component residuals are ambiguous and must be rejected."""
    config = make_config(
        out_dim=2,
        loss_weights={"ra": 1.0, "rb": 1.0},
        pts_weights={"ra": 1.0, "rb": 1.0},
    )
    model = make_model(config, model_cls=TupleTwoComponentIVP)
    batch = make_batch(jax.random.PRNGKey(5))
    with pytest.raises(AssertionError, match="dict"):
        model.compute_residual_losses(model.state.params, model.state, batch)


def test_residual_dict_keys_must_match_pts_weights():
    config = make_config(
        out_dim=2,
        loss_weights={"ra": 1.0, "rb": 1.0},
        pts_weights={"ra": 1.0, "rc": 1.0},  # "rc" instead of "rb"
    )
    model = make_model(config, model_cls=TwoComponentIVP)
    batch = make_batch(jax.random.PRNGKey(5))
    with pytest.raises(AssertionError, match="must match"):
        model.compute_residual_losses(model.state.params, model.state, batch)


def test_pts_weights_applied_by_name():
    """Pseudo-time weights must pair with residual components by name, not
    by dict iteration order."""
    config = make_config(
        out_dim=2,
        loss_weights={"ra": 1.0, "rb": 1.0},
        pts_weights={"ra": 2.0, "rb": 5.0},
        pseudo_time=True,
    )
    model = make_model(config, model_cls=TwoComponentIVP)
    state = model.state
    batch = make_batch(jax.random.PRNGKey(6))

    # prev_params == params, so the pseudo-time shift (w * (sol - sol_prev))
    # is zero and must not perturb the losses regardless of the weights.
    loss_dict = model.compute_residual_losses(
        state.params, state, batch, pseudo_time=True
    )
    np.testing.assert_allclose(loss_dict["rb"], 0.0, atol=1e-6)
    np.testing.assert_allclose(loss_dict["ra"], 9.0, rtol=1e-5)

    # The weight vector itself must follow the r_net dict order ("rb", "ra")
    keys, _ = model._stack_residuals(
        {"rb": jax.numpy.zeros((4,)), "ra": jax.numpy.zeros((4,))}, state
    )
    weights = model._pts_weight_vector(keys, state)
    np.testing.assert_allclose(np.asarray(weights), [5.0, 2.0])


def test_update_loss_weights_matches_reference():
    model = make_model(make_config())
    state = model.state
    batch = make_batch(jax.random.PRNGKey(6))

    ref_weights = model.compute_loss_weights(state, batch)
    new_state = model.update_loss_weights(state, batch)

    momentum = state.momentum
    for key, ref_w in ref_weights.items():
        expected = state.loss_weights[key] * momentum + (1 - momentum) * ref_w
        np.testing.assert_allclose(
            np.asarray(new_state.loss_weights[key]), np.asarray(expected), rtol=1e-4
        )


def test_update_pts_weights_matches_reference():
    model = make_model(make_config(pseudo_time=True))
    init_state = model.state
    batch = make_batch(jax.random.PRNGKey(7))

    # A couple of steps so params != prev_params != init params
    state, _, _ = model.step(init_state, batch)
    state, _, _ = model.step(state, batch)

    ref = model.compute_pts_weights(state, init_state, batch)["res"]
    new_state = model.update_pts_weights(state, init_state, batch)

    pts_momentum = state.pts_momentum
    expected = state.pts_weights["res"] * pts_momentum + (1 - pts_momentum) * ref
    got = new_state.pts_weights["res"]

    assert np.isfinite(got) and got > 0
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-4)


def test_pseudo_time_pairs_residuals_with_their_solutions():
    """The pseudo-time damping must pair each residual with ITS solution
    component. JAX sorts dict keys when flattening pytrees (residual rows
    would come back alphabetical while neural_net outputs are positional) —
    regression test for the misalignment that broke the 3-component
    Navier-Stokes examples (continuity residual damped toward u-velocity)."""
    import jax.numpy as jnp

    config = make_config(
        out_dim=2,
        loss_weights={"res": 1.0},
        pts_weights={"rb": 1.0, "ra": 1.0},
    )
    from helpers import TwoComponentIVP

    model = make_model(config, model_cls=TwoComponentIVP)

    # r_net declares {"rb": ..., "ra": ...}: declaration order must be
    # recovered even though JAX would sort the keys to ["ra", "rb"]
    assert model._res_key_order == ("rb", "ra")

    batch = make_batch(jax.random.PRNGKey(0), n=8)
    coords = tuple(batch[:, i] for i in range(batch.shape[1]))
    keys, res = model._stack_residuals(
        model.r_pred_fn(model.state.params, *coords), model.state
    )
    assert keys == ["rb", "ra"]
    # rb is identically 0 and ra identically 3 -> rows must not be swapped
    np.testing.assert_allclose(np.asarray(res[0]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(res[1]), 3.0, atol=1e-6)

    # positional pairing: first neural_net output (u) pairs with "rb"
    u = jnp.ones(4) * 10.0
    v = jnp.ones(4) * 20.0
    mat = model._sols_matrix((u, v), keys)
    np.testing.assert_array_equal(np.asarray(mat[0]), np.asarray(u))  # rb -> u
    np.testing.assert_array_equal(np.asarray(mat[1]), np.asarray(v))  # ra -> v

    # dict-valued sols are paired by key
    mat = model._sols_matrix({"ra": v, "rb": u}, keys)
    np.testing.assert_array_equal(np.asarray(mat[0]), np.asarray(u))
    np.testing.assert_array_equal(np.asarray(mat[1]), np.asarray(v))

    # component-count mismatches fail loudly
    with pytest.raises(AssertionError):
        model._sols_matrix((u,), keys)
