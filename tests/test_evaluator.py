"""Evaluator tests: every logging path works and the sharded evaluation
helpers (raw residual losses, gradient norms, causal weights) match plain
single-device computations on the full batch."""

import jax
import numpy as np
from jax import jacrev

from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import flatten_pytree

from helpers import make_batch, make_config, make_model
from test_data_parallel import StressIVP, stress_batch, stress_config


def full_config():
    config = stress_config()
    config.logging.log_lr = True
    config.logging.log_losses = True
    config.logging.log_raw_losses = True
    config.logging.log_loss_weights = True
    config.logging.log_pts_weights = True
    config.logging.log_grads = True
    return config


def test_evaluator_logs_all_metrics():
    """All logging flags on, dict batch, multi-component model: every metric
    is present and finite."""
    config = full_config()
    model = make_model(config, model_cls=StressIVP)
    evaluator = BaseEvaluator(config)

    batch = stress_batch(jax.random.PRNGKey(0))
    state, loss, loss_dict = model.step(model.state, batch)

    log_dict = evaluator(model, state, loss_dict, batch)

    expected_keys = {
        "lr",
        # losses
        "loss/u_ic", "loss/v_ic", "loss/u_res", "loss/v_res",
        # raw (unweighted, non-causal) residual losses
        "raw_loss/u_res", "raw_loss/v_res",
        # adaptive weights
        "weights/u_ic", "weights/v_ic", "weights/u_res", "weights/v_res",
        "pts_weights/u", "pts_weights/v",
        # gradient norms
        "grads/u_ic", "grads/v_ic", "grads/u_res", "grads/v_res",
    }
    assert expected_keys <= set(log_dict.keys())
    for key, value in log_dict.items():
        assert np.all(np.isfinite(np.asarray(value))), key


def test_raw_losses_match_unsharded_reference():
    config = full_config()
    model = make_model(config, model_cls=StressIVP)
    state = model.state
    res = make_batch(jax.random.PRNGKey(1))

    sharded = model.compute_raw_residual_losses(state.params, state, res)
    reference = model.compute_residual_losses(state.params, state, res)

    for key in reference:
        np.testing.assert_allclose(
            float(sharded[key]), float(reference[key]), rtol=1e-5, err_msg=key
        )


def test_grad_norms_match_unsharded_reference():
    config = full_config()
    model = make_model(config, model_cls=StressIVP)
    state = model.state
    batch = stress_batch(jax.random.PRNGKey(2))

    sharded = model.compute_grad_norms(state, batch)

    grads = jacrev(model.losses)(state.params, state, batch)
    for key, grad in grads.items():
        reference = np.linalg.norm(np.asarray(flatten_pytree(grad)))
        np.testing.assert_allclose(
            float(sharded[key]), reference, rtol=1e-4, err_msg=key
        )


def test_causal_weights_match_unsharded_reference():
    config = full_config()
    model = make_model(config, model_cls=StressIVP)
    state = model.state
    res = make_batch(jax.random.PRNGKey(3))

    sharded = np.asarray(model.compute_causal_weights(state, res))

    # plain single-device computation of the same gates
    residuals = model._causal_residuals(
        state.params, state, res, config.pseudo_time.enabled
    )
    chunk_loss = np.asarray(residuals).reshape(
        residuals.shape[0], config.causal.num_chunks, -1
    )
    chunk_loss = (chunk_loss**2).mean(axis=2)
    cumulative = np.concatenate(
        [np.zeros((chunk_loss.shape[0], 1)), np.cumsum(chunk_loss, axis=1)[:, :-1]],
        axis=1,
    )
    reference = np.exp(-config.causal.tol * cumulative).min(axis=0)

    assert sharded.shape == (config.causal.num_chunks,)
    np.testing.assert_allclose(sharded, reference, rtol=1e-5)


def test_evaluator_accepts_plain_array_batch():
    """Single-sampler examples pass a raw array as the batch."""
    config = make_config()
    config.logging.log_raw_losses = True
    config.logging.log_grads = True
    model = make_model(config)
    evaluator = BaseEvaluator(config)

    batch = make_batch(jax.random.PRNGKey(4))
    state, loss, loss_dict = model.step(model.state, batch)
    log_dict = evaluator(model, state, loss_dict, batch)

    assert "raw_loss/res" in log_dict
    assert "grads/res" in log_dict and "grads/ics" in log_dict
