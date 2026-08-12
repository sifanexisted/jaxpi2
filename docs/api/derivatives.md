# jaxpi.derivatives

Forward-mode derivative helpers for PDE residuals.

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

## `value_and_jacfwd()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/derivatives.py#L43' target='_blank'>[source]</a>

```python
value_and_jacfwd(f, argnums)
```

Values of `f` and first derivatives wrt the scalar args `argnums`.

Returns `g(*args) -> (values, derivs)` where `derivs[j]` has the same
structure as `values` and holds d(values)/d(args[argnums[j]]). The
function is traced once (`jax.linearize`); each derivative costs one
JVP evaluation of the linearization.

Example (2D NS): `(u, v, p), (d_t, d_x, d_y) = value_and_jacfwd(
net, (1, 2, 3))(params, t, x, y)`, with `d_x = (u_x, v_x, p_x)`.

## `jacfwd_scalar_args()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/derivatives.py#L76' target='_blank'>[source]</a>

```python
jacfwd_scalar_args(f, argnums)
```

First derivatives only (same layout as `value_and_jacfwd`[1]).

## `hessian_diag_fwd()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/derivatives.py#L86' target='_blank'>[source]</a>

```python
hessian_diag_fwd(f, argnums)
```

Pure second derivatives d^2 f / d args[i]^2 for each i in `argnums`.

Forward-over-forward: for each coordinate, a JVP of a JVP with the
same unit tangent. Only the requested diagonal entries are computed —
a Laplacian over n coordinates costs n nested JVPs, with none of the
cross terms `jax.hessian` would build.

Returns `g(*args) -> diag` with `diag[j]` structured like `f`'s
output, holding d^2(outputs)/d(args[argnums[j]])^2.

## `derivatives_fwd_1d()` <a class='source-link' href='https://github.com/sifanexisted/jaxpi2/blob/main/jaxpi/derivatives.py#L116' target='_blank'>[source]</a>

```python
derivatives_fwd_1d(f, argnum, order)
```

Successive derivatives (f', f'', ..., f^(order)) wrt one scalar arg.

Nested forward-mode JVPs building one tower: level k differentiates
the whole (value, d1, ..., d_{k-1}) stack once more, so all orders
come out of a single nested evaluation. Replaces `grad`-of-`grad`
chains whose nested tapes grow with order.
Returns `g(*args) -> (d1, d2, ..., d_order)`.
