from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jaxpi.models import create_lr_schedule, create_optimizer
from jaxpi.utils import create_update_scheduler, flatten_pytree, get_eval_params

from helpers import make_config


def test_flatten_pytree():
    tree = {"a": jnp.ones((2, 2)), "b": jnp.zeros(3)}
    flat = flatten_pytree(tree)
    assert flat.shape == (7,)


def test_create_update_scheduler():
    should_update = create_update_scheduler(every=100, start=50)
    assert not should_update(0)
    assert not should_update(49)
    assert should_update(50)
    assert not should_update(51)
    assert should_update(150)
    assert should_update(250)


def test_get_eval_params_without_schedule_free():
    params = {"w": jnp.ones(3)}
    state = SimpleNamespace(params=params, opt_state=optax.adam(1e-3).init(params))
    assert get_eval_params(state, schedule_free=False) is params


def test_get_eval_params_with_schedule_free():
    cfg = make_config(schedule_free=True).optim
    lr = create_lr_schedule(cfg)
    tx = create_optimizer(cfg, lr)

    params = {"w": jnp.ones(3)}
    opt_state = tx.init(params)
    for _ in range(3):
        grads = {"w": jnp.full(3, 0.1)}
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    eval_params = get_eval_params(
        SimpleNamespace(params=params, opt_state=opt_state), schedule_free=True
    )
    assert eval_params["w"].shape == (3,)
    assert np.all(np.isfinite(np.asarray(eval_params["w"])))

    # Must match optax's own eval-params computation on the wrapped state
    sf_state = opt_state[1]  # (clip_by_global_norm, schedule_free)
    expected = optax.contrib.schedule_free_eval_params(sf_state, params)
    np.testing.assert_allclose(np.asarray(eval_params["w"]), np.asarray(expected["w"]))


def test_get_eval_params_missing_schedule_free_state():
    params = {"w": jnp.ones(3)}
    state = SimpleNamespace(params=params, opt_state=optax.adam(1e-3).init(params))
    with pytest.raises(ValueError, match="schedule-free"):
        get_eval_params(state, schedule_free=True)
