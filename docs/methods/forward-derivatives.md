# Forward-Mode Residual Derivatives

::: info TL;DR
PDE residuals differentiate the network with respect to a **handful of scalar
coordinates** — the opposite regime from backprop. JAXPI computes them with forward-mode
JVP sweeps (`jaxpi.derivatives`): values identical to the reverse-mode idioms to floating
point rounding, residual evaluation **1.4–3× faster** on multi-output models, and no tape
in memory. The parameter gradient of the loss is unaffected and stays reverse-mode.
:::

## The problem: the residual stack is the inner loop

A Navier–Stokes-style residual needs a lot of derivatives of the network
$f_\theta : (t, x, y) \mapsto (u, v, p)$ at every collocation point of every training step:
the full first-derivative Jacobian plus the velocity Laplacians,

$$
J \;=\; \frac{\partial(u, v, p)}{\partial(t, x, y)} \in \mathbb{R}^{3\times3},
\qquad
u_{xx},\; u_{yy},\; v_{xx},\; v_{yy}.
$$

The idiomatic JAX implementation — one `jacrev` for the Jacobian, one `jax.hessian` per
velocity component — does redundant work in exactly this regime:

- `jacrev` runs **one taped backward pass per output**: forward pass, store every
  intermediate, transpose sweep. Three outputs, three tapes.
- `jax.hessian` per component builds the **full** spatial Hessian, including the
  cross terms $u_{xy}$, $v_{xy}$ that a Laplacian never uses — and it pays this
  **per component**, so a 3D model burns nine taped Hessian columns to keep six numbers.

None of this is wrong — the values are exact — but the shape of the computation is wrong
for the shape of the problem.

## Forward vs reverse: pick the mode by shape

The network is a composition $f = f_L \circ \cdots \circ f_1$, so its Jacobian is the
matrix product $J = J_L \cdots J_1$. Autodiff never forms the factors; it applies them to
vectors, in one of two orders:

$$
\textbf{JVP (forward):}\quad \dot z_{k+1} = J_k\, \dot z_k
\qquad\qquad
\textbf{VJP (reverse):}\quad \bar z_k = J_k^{\top} \bar z_{k+1}
$$

One JVP costs about two forward passes, stores **nothing**, and yields $J s$ — the
derivative of *every output* in one input direction $s$. One VJP yields $w^{\top} J$ — the
gradient of *one output* with respect to every input — but must record a tape and run a
transpose sweep. A full Jacobian therefore costs $n_{\text{in}}$ JVPs or
$n_{\text{out}}$ VJPs:

| map | shape | right mode |
|---|---|---|
| loss w.r.t. parameters | $10^6 \to 1$ | reverse (backprop) — unchanged |
| residual w.r.t. coordinates | $3\text{–}4 \to 2\text{–}5$ | **forward** — same pass count, no tape, no transpose |

Both orders evaluate the same chain-rule product, and matrix multiplication is
associative — so the values are **identical up to float rounding**. This is a pure
performance and memory decision, never an accuracy trade-off.

For second derivatives, a JVP of a JVP gives a second-order directional derivative,
$\partial_r \partial_s f = r^{\top}(\nabla^2 f)\,s$; seeding both levels with the same
basis vector $e_i$ reads off exactly the diagonal entry $\partial^2 f / \partial c_i^2$.
Two properties make this strictly better than `jax.hessian` here:

- **only the diagonal** is computed — no cross terms;
- each sweep carries **all output components at once**, so the cost scales with the
  number of *coordinates*, not outputs. The old per-component idiom paid
  $m \times$ full Hessians; the forward version pays $d_s$ nested sweeps. For the 3D
  Taylor–Green model that turns nine taped Hessians into three tape-free sweeps.

## Design decisions

`jaxpi.derivatives` deliberately stays small — three primitives that mirror how residuals
are actually written:

- **`value_and_jacfwd(f, argnums)`** traces $f$ once with `jax.linearize` and evaluates
  one JVP per coordinate, returning values *and* all first derivatives — the common
  "call the network, then differentiate it" double evaluation disappears.
- **`hessian_diag_fwd(f, argnums)`** is the forward-over-forward diagonal described above.
- **`derivatives_fwd_1d(f, argnum, order)`** builds one nested tower for successive
  derivatives along a single coordinate.
- Helpers take **scalar coordinate arguments**, matching the per-point `r_net` signature;
  batching stays outside in `vmap`, exactly as before.

Two variants were benchmarked and rejected:

- *Vectorizing the JVPs over the coordinate basis* (a `vmap` over tangents instead of a
  Python loop) looks elegant but composes badly with the outer batch `vmap`: Taylor–Green
  regressed from 9.7 ms to 13.4 ms. XLA fuses the sequential form better; it stayed.
- *Migrating everything unconditionally.* Two residual families are better off as they
  were, and keep their original code: **first-order scalar residuals** (`advection`,
  `inviscid_burgers`), where reverse mode needs one pass against forward's two and
  forward measured 0.72×; and **high-order 1D chains** (`kdv`, `ks`), where nested JVP
  towers grow like $2^k$ and Taylor-mode `jet` already computes the whole derivative
  tower in a single pass.

During training the residual sits inside the loss, whose parameter gradient is taken by
reverse mode — so the full computation is reverse-over-forward. JAX transposes JVPs
cheaply; the measured step-time gains track the residual-level gains.

## Results

Residual evaluation (`vmap`-ed `r_net`, 8192 points, one GPU), reverse-mode idiom → 
forward-mode helpers:

| example | before | after | speedup |
|---|---|---|---|
| rayleigh_taylor (4 outputs, 2D) | 11.1 ms | 3.7 ms | **3.0×** |
| taylor_green (4 outputs, 3D) | 24.8 ms | 9.7 ms | **2.5×** |
| lid_driven_cavity / bfs_flow | 3.2 ms | 1.5 ms | 2.1× |
| kolmogorov_flow_Re1e6 | 15.1 ms | 7.3 ms | 2.1× |
| taylor_green multi-stage | 49.3 ms | 24.5 ms | 2.0× |
| kolmogorov_flow | 6.8 ms | 3.7 ms | 1.9× |
| ginzburg_landau / gray_scott | | | 1.7× / 1.4× |
| wave / burgers / allen_cahn | | | 1.05–1.12× |
| kdv / ks / sod_shock_tube / advection | | | unchanged (already optimal) |

The ranking follows the theory: gains grow with output count and spatial dimension,
because those multiply the redundant reverse passes the old idiom paid.

## In JAXPI

A 2D Navier–Stokes residual in the forward-mode idiom:

```python
from jaxpi.derivatives import hessian_diag_fwd, value_and_jacfwd

def r_net(self, params, t, x, y):
    (u, v, p), (d_t, d_x, d_y) = value_and_jacfwd(
        self.neural_net, (1, 2, 3))(params, t, x, y)
    u_t, v_t, _ = d_t
    u_x, v_x, p_x = d_x
    u_y, v_y, p_y = d_y

    (u_xx, v_xx, _), (u_yy, v_yy, _) = hessian_diag_fwd(
        self.neural_net, (2, 3))(params, t, x, y)
    ...
```

Every example migration was verified against fixtures captured from the pre-migration
code (`tests/residual_equivalence.py --capture / --verify`): all 17 reproduce their
original residuals to $\le 5\times10^{-15}$ relative in float64, and
`tests/test_derivatives.py` pins the helpers against closed-form derivatives and the
reverse-mode references. The `--bench` mode reproduces the table above. See the
[jaxpi.derivatives API reference](/api/derivatives).
