"""Forward-mode derivative helpers for PDE residuals.

PINN residuals differentiate the network with respect to a handful of
scalar coordinates — a few-inputs / few-outputs problem for which
forward-mode AD is the right shape: one JVP sweep per coordinate yields
the derivative of every output, with no tape and no transpose pass.
(The parameter gradient of the loss is unaffected and stays reverse-mode.)

These helpers cover the two patterns residuals need:

  - `value_and_jacfwd`: values and all first derivatives, one traced
    forward evaluation (via `jax.linearize`) plus one JVP per coordinate.
  - `hessian_diag_fwd`: pure second derivatives d^2 f / d c_i^2 by
    forward-over-forward JVPs — only the diagonal entries a Laplacian
    needs, never the cross terms that `jax.hessian` would also compute.
  - `derivatives_fwd_1d`: successive derivatives along one coordinate
    (for KdV / KS style high-order 1D operators), by nested JVPs.

All helpers assume the differentiated arguments are scalars (the
standard per-collocation-point residual signature; batching is done
outside with `vmap`, exactly as before). Values agree with the
reverse-mode idioms (`jacrev`, `hessian`, nested `grad`) to floating
point rounding: both orderings evaluate the same chain-rule product.
"""

from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["value_and_jacfwd", "jacfwd_scalar_args", "hessian_diag_fwd",
           "derivatives_fwd_1d"]


def _substitute(args, argnums, values):
    """Return `args` with positions `argnums` replaced by `values`."""
    args = list(args)
    for i, v in zip(argnums, values):
        args[i] = v
    return tuple(args)


def value_and_jacfwd(f, argnums):
    """Values of `f` and first derivatives wrt the scalar args `argnums`.

    Returns `g(*args) -> (values, derivs)` where `derivs[j]` has the same
    structure as `values` and holds d(values)/d(args[argnums[j]]). The
    function is traced once (`jax.linearize`); each derivative costs one
    JVP evaluation of the linearization.

    Example (2D NS): `(u, v, p), (d_t, d_x, d_y) = value_and_jacfwd(
    net, (1, 2, 3))(params, t, x, y)`, with `d_x = (u_x, v_x, p_x)`.
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def wrapped(*args):
        primals = tuple(args[i] for i in argnums)

        def f_of(*sel):
            return f(*_substitute(args, argnums, sel))

        values, jvp_fn = jax.linearize(f_of, *primals)
        derivs = []
        for j in range(len(argnums)):
            tangents = tuple(
                jnp.ones_like(primals[k]) if k == j else jnp.zeros_like(primals[k])
                for k in range(len(argnums))
            )
            derivs.append(jvp_fn(*tangents))
        return values, tuple(derivs)

    return wrapped


def jacfwd_scalar_args(f, argnums):
    """First derivatives only (same layout as `value_and_jacfwd`[1])."""
    fn = value_and_jacfwd(f, argnums)

    def wrapped(*args):
        return fn(*args)[1]

    return wrapped


def hessian_diag_fwd(f, argnums):
    """Pure second derivatives d^2 f / d args[i]^2 for each i in `argnums`.

    Forward-over-forward: for each coordinate, a JVP of a JVP with the
    same unit tangent. Only the requested diagonal entries are computed —
    a Laplacian over n coordinates costs n nested JVPs, with none of the
    cross terms `jax.hessian` would build.

    Returns `g(*args) -> diag` with `diag[j]` structured like `f`'s
    output, holding d^2(outputs)/d(args[argnums[j]])^2.
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def wrapped(*args):
        out = []
        for i in argnums:

            def first_deriv(c, i=i):
                inner = lambda ci: f(*_substitute(args, (i,), (ci,)))
                return jax.jvp(inner, (c,), (jnp.ones_like(c),))[1]

            _, second = jax.jvp(first_deriv, (args[i],),
                                (jnp.ones_like(args[i]),))
            out.append(second)
        return tuple(out)

    return wrapped


def derivatives_fwd_1d(f, argnum, order):
    """Successive derivatives (f', f'', ..., f^(order)) wrt one scalar arg.

    Nested forward-mode JVPs building one tower: level k differentiates
    the whole (value, d1, ..., d_{k-1}) stack once more, so all orders
    come out of a single nested evaluation. Replaces `grad`-of-`grad`
    chains whose nested tapes grow with order.
    Returns `g(*args) -> (d1, d2, ..., d_order)`.
    """

    def wrapped(*args):
        def tower(k):
            """c -> (value, d1, ..., dk)."""
            if k == 0:
                return lambda c: (f(*_substitute(args, (argnum,), (c,))),)
            lower = tower(k - 1)

            def level(c):
                vals, tangs = jax.jvp(lower, (c,), (jnp.ones_like(c),))
                # tangs = (d1, ..., dk) of the lower stack
                return vals + (tangs[-1],)

            return level

        return tower(order)(args[argnum])[1:]

    return wrapped
