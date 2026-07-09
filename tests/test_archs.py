import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxpi import archs
from jaxpi.models import create_arch

from helpers import make_config


@pytest.mark.parametrize("name", ["Mlp", "ModifiedMlp", "PirateNet"])
def test_arch_output_shape(name):
    cfg = make_config(out_dim=3).arch
    cfg.arch_name = name
    if name == "PirateNet":
        # PirateNet's residual blocks require embedded input == hidden_dim
        cfg.fourier_emb = {"embed_scale": 1.0, "embed_dim": cfg.hidden_dim}
    arch = create_arch(cfg)

    x = jnp.ones(2)
    params = arch.init(random.PRNGKey(0), x)
    y = arch.apply(params, x)

    assert y.shape == (3,)
    assert jnp.all(jnp.isfinite(y))


def test_piratenet_without_matching_embed_dim_raises():
    cfg = make_config().arch
    cfg.arch_name = "PirateNet"
    arch = create_arch(cfg)
    with pytest.raises(AssertionError, match="hidden_dim"):
        arch.init(random.PRNGKey(0), jnp.ones(2))


def test_create_arch_case_insensitive():
    cfg = make_config().arch
    cfg.arch_name = "pIrAtEnEt"
    assert isinstance(create_arch(cfg), archs.PirateNet)


def test_unknown_activation_raises():
    cfg = make_config().arch
    cfg.activation = "softplus2"
    arch = create_arch(cfg)
    with pytest.raises(NotImplementedError):
        arch.init(random.PRNGKey(0), jnp.ones(2))


def test_fourier_embs_output_dim():
    module = archs.FourierEmbs(embed_scale=1.0, embed_dim=32)
    x = jnp.ones(2)
    params = module.init(random.PRNGKey(0), x)
    y = module.apply(params, x)
    assert y.shape == (32,)


def test_fourier_embs_odd_dim_raises():
    module = archs.FourierEmbs(embed_scale=1.0, embed_dim=33)
    with pytest.raises(AssertionError, match="even"):
        module.init(random.PRNGKey(0), jnp.ones(2))


def test_period_embs_periodicity():
    freq = 2.0
    module = archs.PeriodEmbs(period=(freq,), axis=(0,), trainable=(False,))
    x = jnp.array([0.3, 1.7])
    params = module.init(random.PRNGKey(0), x)

    y = module.apply(params, x)
    y_shifted = module.apply(params, x.at[0].add(2 * jnp.pi / freq))

    # cos/sin pair for axis 0 plus the untouched axis 1
    assert y.shape == (3,)
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_shifted), atol=1e-5)


def test_arch_with_embeddings():
    cfg = make_config().arch
    cfg.arch_name = "Mlp"
    cfg.periodicity = {"period": (2.0,), "axis": (1,), "trainable": (False,)}
    cfg.fourier_emb = {"embed_scale": 1.0, "embed_dim": 16}
    arch = create_arch(cfg)

    x = jnp.ones(2)
    params = arch.init(random.PRNGKey(0), x)
    y = arch.apply(params, x)
    assert y.shape == (1,)
    assert jnp.all(jnp.isfinite(y))
