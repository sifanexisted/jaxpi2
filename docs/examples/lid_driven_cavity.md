# Lid-driven Cavity

Steady incompressible flow in a square cavity driven by a moving lid — a boundary-value
problem (no time variable):

$$
(\mathbf{u} \cdot \nabla)\,\mathbf{u} + \nabla p - \nu \nabla^2 \mathbf{u} = 0,
\qquad \nabla \cdot \mathbf{u} = 0 \quad \text{in } [0,1]^2 .
$$

## Problem setup

The domain is the unit square $(x, y) \in [0,1]^2$. The top lid slides tangentially with
$u = 1,\; v = 0$, while the left, right, and bottom walls are no-slip ($u = v = 0$). The
Reynolds number is $\mathrm{Re} = UL/\nu$ with lid speed $U = 1$ and cavity size $L = 1$;
the showcase runs at $\mathrm{Re} = 5000$ ($\nu = 1/5000$), where a strong primary vortex
fills the cavity and counter-rotating eddies develop in the bottom corners.

<figure class="example-figure">

![Lid-driven cavity setup](/jaxpi2/setup/ldc_setup.svg)

<figcaption>Geometry and boundary conditions. The lid velocity is discontinuous at the two
top corners, which concentrates error and makes high-Re training stiff.</figcaption>
</figure>

Both boundary conditions enter the loss as sampled point sets: a `u_bc` term fitting the
lid profile (including the discontinuous corners) and a `v_bc` term enforcing zero normal
flow, alongside the three momentum/continuity residuals — five loss terms balanced by
grad-norm weighting.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>0.043</strong></span>
  <span>recipe <strong>adaptive pseudo-time</strong></span>
  <span>Re 5000, 50k steps, single GPU</span>
</div>

Direct training at Re 5000 stalls: the baseline plateaus at a relative L2 error of
**0.64** in the velocity magnitude, trapped in a low-circulation spurious solution
(swapping SOAP for Adam changes nothing — 0.68). Adaptive
[pseudo-time stepping](/methods/pseudo-time) acts as an artificial-time continuation
toward the steady state and reaches **0.043** — a 15× improvement and the difference
between a wrong flow field and a quantitatively correct one. The comparison is sharp:
constant pseudo-time weights do *not* help here (0.84); the adaptive step-size estimate is
what tracks the slowly developing circulation.

<figure class="example-figure">

![Lid-driven cavity prediction vs reference](/jaxpi2/results/lid_driven_cavity_pred.png)

<figcaption>Reference and predicted velocity magnitude at Re 5000, and the absolute error.</figcaption>
</figure>

<figure class="example-figure">

![Lid-driven cavity convergence](/jaxpi2/results/lid_driven_cavity_convergence.png)

<figcaption>Training losses and relative L2 error of the showcase (adaptive pseudo-time) run.</figcaption>
</figure>

## Run

```bash
cd examples/lid_driven_cavity
python3 main.py --config=configs/pseudo_time.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | Modified MLP baseline |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping (showcase) |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Steady problems subclass `ForwardBVP` — same trainer, no causal weighting.
- Reference data ships for Re ∈ {100, 400, 1000, 1600, 3200, 5000}
  (`--config.Re=...`).
