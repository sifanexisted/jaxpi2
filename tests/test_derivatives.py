"""Unit tests for jaxpi.derivatives (forward-mode residual helpers).

Each helper is checked two ways, in float64:
  1. against closed-form derivatives of an analytic function, and
  2. against the reverse-mode idioms it replaces (jacrev / hessian /
     nested grad) on both the analytic function and a small MLP,
including composition under vmap (how the helpers are used in r_nets).

Run:  PYTHONPATH=. pytest tests/test_derivatives.py -q
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad, hessian, jacrev

from jaxpi.derivatives import (derivatives_fwd_1d, hessian_diag_fwd,
                               jacfwd_scalar_args, value_and_jacfwd)

ATOL = 1e-10


def analytic(t, x, y):
    """3-in / 3-out with hand-computable derivatives."""
    return (jnp.sin(t * x) * jnp.cos(y),
            t ** 2 + x * y,
            jnp.exp(0.3 * x) * y)


def tiny_mlp():
    rng = np.random.default_rng(0)
    w1 = jnp.asarray(rng.normal(size=(3, 16)) / 3.0)
    b1 = jnp.asarray(rng.normal(size=16) * 0.1)
    w2 = jnp.asarray(rng.normal(size=(16, 2)) / 4.0)

    def f(params, t, x, y):
        h = jnp.tanh(jnp.stack([t, x, y]) @ params["w1"] + params["b1"])
        out = h @ params["w2"]
        return out[0], out[1]

    return f, {"w1": w1, "b1": b1, "w2": w2}


TXY = (0.7, -0.4, 1.3)


class TestValueAndJacfwd:
    def test_analytic_closed_form(self):
        t, x, y = TXY
        (f0, f1, f2), (d_t, d_x, d_y) = value_and_jacfwd(
            analytic, (0, 1, 2))(*TXY)

        np.testing.assert_allclose(f0, np.sin(t * x) * np.cos(y), atol=ATOL)
        # d/dt
        np.testing.assert_allclose(d_t[0], x * np.cos(t * x) * np.cos(y), atol=ATOL)
        np.testing.assert_allclose(d_t[1], 2 * t, atol=ATOL)
        np.testing.assert_allclose(d_t[2], 0.0, atol=ATOL)
        # d/dx
        np.testing.assert_allclose(d_x[0], t * np.cos(t * x) * np.cos(y), atol=ATOL)
        np.testing.assert_allclose(d_x[1], y, atol=ATOL)
        np.testing.assert_allclose(d_x[2], 0.3 * np.exp(0.3 * x) * y, atol=ATOL)
        # d/dy
        np.testing.assert_allclose(d_y[0], -np.sin(t * x) * np.sin(y), atol=ATOL)
        np.testing.assert_allclose(d_y[1], x, atol=ATOL)
        np.testing.assert_allclose(d_y[2], np.exp(0.3 * x), atol=ATOL)

    def test_matches_jacrev_on_mlp(self):
        f, params = tiny_mlp()
        values, (d_t, d_x, d_y) = value_and_jacfwd(f, (1, 2, 3))(params, *TXY)
        ref_vals = f(params, *TXY)
        ref = jacrev(f, argnums=(1, 2, 3))(params, *TXY)
        for k in range(2):  # output component
            np.testing.assert_allclose(values[k], ref_vals[k], atol=ATOL)
            for j, d in enumerate((d_t, d_x, d_y)):  # coordinate
                np.testing.assert_allclose(d[k], ref[k][j], atol=ATOL)

    def test_int_argnums_and_subset(self):
        f, params = tiny_mlp()
        _, ((u_t, v_t),) = value_and_jacfwd(f, (1,))(params, *TXY)
        u_t_ref = grad(lambda p, t, x, y: f(p, t, x, y)[0], argnums=1)(
            params, *TXY)
        v_t_ref = grad(lambda p, t, x, y: f(p, t, x, y)[1], argnums=1)(
            params, *TXY)
        np.testing.assert_allclose(u_t, u_t_ref, atol=ATOL)
        np.testing.assert_allclose(v_t, v_t_ref, atol=ATOL)

    def test_under_vmap(self):
        f, params = tiny_mlp()
        pts = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, (32, 3)))
        fn = jax.vmap(
            lambda t, x, y: value_and_jacfwd(f, (1, 2, 3))(params, t, x, y),
            (0, 0, 0))
        (u, v), (d_t, d_x, d_y) = fn(pts[:, 0], pts[:, 1], pts[:, 2])
        ref = jax.vmap(
            lambda t, x, y: jacrev(f, argnums=(1, 2, 3))(params, t, x, y),
            (0, 0, 0))(pts[:, 0], pts[:, 1], pts[:, 2])
        np.testing.assert_allclose(d_x[0], ref[0][1], atol=ATOL)
        np.testing.assert_allclose(d_y[1], ref[1][2], atol=ATOL)
        assert u.shape == (32,)


class TestHessianDiagFwd:
    def test_analytic_closed_form(self):
        t, x, y = TXY
        (d2x, d2y) = hessian_diag_fwd(analytic, (1, 2))(*TXY)
        np.testing.assert_allclose(
            d2x[0], -t ** 2 * np.sin(t * x) * np.cos(y), atol=ATOL)
        np.testing.assert_allclose(d2x[1], 0.0, atol=ATOL)
        np.testing.assert_allclose(
            d2x[2], 0.09 * np.exp(0.3 * x) * y, atol=ATOL)
        np.testing.assert_allclose(
            d2y[0], -np.sin(t * x) * np.cos(y), atol=ATOL)
        np.testing.assert_allclose(d2y[1], 0.0, atol=ATOL)
        np.testing.assert_allclose(d2y[2], 0.0, atol=ATOL)

    def test_matches_hessian_on_mlp(self):
        f, params = tiny_mlp()
        d2x, d2y = hessian_diag_fwd(f, (2, 3))(params, *TXY)
        for k in range(2):
            h = hessian(lambda p, t, x, y, k=k: f(p, t, x, y)[k],
                        argnums=(2, 3))(params, *TXY)
            np.testing.assert_allclose(d2x[k], h[0][0], atol=ATOL)
            np.testing.assert_allclose(d2y[k], h[1][1], atol=ATOL)

    def test_scalar_output_single_coord(self):
        g = lambda t, x: jnp.sin(3.0 * x) * t
        (u_xx,) = hessian_diag_fwd(g, (1,))(0.5, 0.8)
        np.testing.assert_allclose(
            u_xx, -9.0 * np.sin(3.0 * 0.8) * 0.5, atol=ATOL)


class TestJacfwdScalarArgs:
    def test_matches_value_and_jacfwd(self):
        f, params = tiny_mlp()
        derivs = jacfwd_scalar_args(f, (2, 3))(params, *TXY)
        _, ref = value_and_jacfwd(f, (2, 3))(params, *TXY)
        for d, r in zip(derivs, ref):
            np.testing.assert_allclose(d, r, atol=ATOL)


class TestDerivativesFwd1d:
    def test_analytic_high_order(self):
        g = lambda t, x: t * jnp.sin(2.0 * x)
        t, x = 0.9, 0.3
        d1, d2, d3, d4 = derivatives_fwd_1d(g, 1, 4)(t, x)
        s, c = np.sin(2 * x), np.cos(2 * x)
        np.testing.assert_allclose(d1, t * 2 * c, atol=ATOL)
        np.testing.assert_allclose(d2, -t * 4 * s, atol=ATOL)
        np.testing.assert_allclose(d3, -t * 8 * c, atol=ATOL)
        np.testing.assert_allclose(d4, t * 16 * s, atol=ATOL)

    def test_matches_nested_grad_on_mlp(self):
        f, params = tiny_mlp()
        u = lambda t, x, y: f(params, t, x, y)[0]
        d1, d2, d3 = derivatives_fwd_1d(u, 1, 3)(*TXY)
        g1 = grad(u, argnums=1)
        g2 = grad(g1, argnums=1)
        g3 = grad(g2, argnums=1)
        np.testing.assert_allclose(d1, g1(*TXY), atol=ATOL)
        np.testing.assert_allclose(d2, g2(*TXY), atol=ATOL)
        np.testing.assert_allclose(d3, g3(*TXY), atol=ATOL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
