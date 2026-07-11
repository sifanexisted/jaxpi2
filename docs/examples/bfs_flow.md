# Backward-facing Step

Steady channel flow over a backward-facing step at $\mathrm{Re} = 800$ — a boundary-value
problem with separation and reattachment:

$$
(\mathbf{u} \cdot \nabla)\,\mathbf{u} + \nabla p - \nu \nabla^2 \mathbf{u} = 0,
\qquad \nabla \cdot \mathbf{u} = 0 .
$$

## Problem setup

The channel spans $x \in [0, 15]$, $y \in [-\tfrac12, \tfrac12]$. Flow enters only through
the **upper half** of the inlet plane: the parabolic profile
$u(y) = 24\,y\,(\tfrac12 - y)$, $v = 0$ is positive for $y \in (0, \tfrac12)$ (peaking at
$u = 1.5$ at $y = \tfrac14$) and zero below, so the lower half of the inlet acts as the
vertical face of a backward-facing step of height $\tfrac12$ — an expansion ratio of 2.
The kinematic viscosity is $\nu = 1/800$, giving $\mathrm{Re} = 800$. Downstream of the
step the flow separates, forms a recirculation bubble along the lower wall, and reattaches
several step heights downstream — the quantity PINN training must get right.

<figure class="example-figure">

![Backward-facing step setup](/jaxpi2/setup/bfs_setup.svg)

<figcaption>Geometry and boundary conditions (streamwise axis compressed; the channel is
15 units long).</figcaption>
</figure>

Boundary conditions, exactly as they enter the loss:

- **Inflow** ($x = 0$): $u = 24\,y\,(\tfrac12 - y)$ for $y > 0$ (zero on the step face),
  $v = 0$ — terms `u_in`, `v_in`.
- **Walls** ($y = \pm\tfrac12$, and the step face): no-slip $u = v = 0$ — terms
  `u_noslip`, `v_noslip`.
- **Outflow** ($x = 15$): $p = 0$ — term `p_out`.

Together with the three momentum/continuity residuals, that is eight loss terms balanced
by grad-norm weighting; collocation points are drawn from the unstructured reference mesh
with `MeshSampler` since the geometry is non-rectangular.

<figure class="example-figure">

![Backward-facing step velocity magnitude](/jaxpi2/gallery/bfs_flow.png)

<figcaption>Reference velocity magnitude: recirculation bubble behind the step.</figcaption>
</figure>

## Results

<div class="result-glance">
  <span>relative L2 error <strong>0.016</strong></span>
  <span>recipe <strong>fixed pseudo-time</strong></span>
  <span>Re 800, 100k steps, single GPU</span>
</div>

The baseline never finds the recirculation bubble: it settles at a relative L2 error of
**0.28** in $u$ (0.83 in $v$), an attached-flow spurious solution. Pseudo-time stepping
recovers the separation, and here the *constant*-weight variant wins: **0.029** in $u$
and 0.106 in $v$ — **0.016** in the velocity magnitude, a ~17× improvement over the
baseline. The adaptive variant helps but lands mid-way (0.15 in $u$) and is
seed-sensitive on this problem; the recirculating flow develops slowly enough that the
adaptive step-size estimate keeps shrinking the damping too early.

<figure class="example-figure">

![Backward-facing step prediction vs reference](/jaxpi2/results/bfs_flow_pred.png)

<figcaption>Reference and predicted velocity magnitude, and the absolute error —
concentrated along the shear layer and reattachment region.</figcaption>
</figure>

<figure class="example-figure">

![Backward-facing step convergence](/jaxpi2/results/bfs_flow_convergence.png)

<figcaption>Training losses and per-variable relative L2 errors of the showcase
(fixed pseudo-time) run.</figcaption>
</figure>

## Run

```bash
cd examples/bfs_flow
python3 main.py --config=configs/fixed_pseudo_time.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet baseline |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights (showcase) |

## Notes

- Reading the `.vtu` reference data requires `pip install pyvista`.
