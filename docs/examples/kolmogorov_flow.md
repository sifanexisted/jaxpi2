# Kolmogorov Flow

2D incompressible Navier–Stokes on a periodic square, driven by a sinusoidal body force —
the flow is unsteady and vortical:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\,\mathbf{u}
  + \nabla p - \nu \nabla^2 \mathbf{u} = \mathbf{f},
\qquad \nabla \cdot \mathbf{u} = 0,
\qquad \mathbf{f} = (2 \sin(4\pi y),\, 0).
$$

## Results

<div class="result-glance">
  <span>vorticity error <strong>4.2e-02</strong></span>
  <span>recipe <strong>4 time windows + fixed pseudo-time</strong></span>
  <span>4 × 100k steps, single GPU</span>
</div>

The vortical dynamics are tracked over the full horizon to a relative vorticity error of
**4.2e-02** (stitched across all four windows), versus **0.59** when trained as a single
global fit — the time-window curriculum is the decisive ingredient for this flow.
Pseudo-time stepping composes cleanly with it: constant damping weights (step size 1.0)
train stably through every window, and an adaptive-weight variant matches it
(4.4e-02 even without windows in the single-window configuration). The animation shows the
prediction staying phase-locked with the reference as vortices merge and stretch; the
error concentrates in the thin filaments between vortices.

<figure class="example-figure">
  <video src="/jaxpi2/results/kolmogorov_flow_pred.mp4" autoplay loop muted playsinline></video>
  <figcaption>Reference, PINN prediction, and absolute error of the vorticity field over
  all four time windows.</figcaption>
</figure>

<figure class="example-figure">

![Kolmogorov flow convergence](/jaxpi2/results/kolmogorov_flow_convergence.png)

<figcaption>Training losses and per-variable errors across the four windows (each window
warm-starts from the previous one's parameters).</figcaption>
</figure>

## Run

```bash
cd examples/kolmogorov_flow
python3 main.py --config=configs/pseudo_time.py \
    --config.pseudo_time.strategy=constant \
    --config.training.num_time_windows=4
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Uses [time-window training](/guide/training-techniques#time-window-curriculum) with
  transfer learning between windows; the vorticity is computed from the velocity network
  by automatic differentiation ($\omega = v_x - u_y$).
- The higher-Reynolds version of this problem is the
  [Kolmogorov flow at Re 10⁶](/examples/kolmogorov_flow_Re1e6).
