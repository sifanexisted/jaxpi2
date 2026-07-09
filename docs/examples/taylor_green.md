# Taylor–Green Vortex

The classic 3D transition-to-turbulence benchmark at $\mathrm{Re} = 1600$: incompressible
Navier–Stokes on the periodic cube $[0, 2\pi]^3$,

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\,\mathbf{u}
  + \nabla p - \nu \nabla^2 \mathbf{u} = 0,
\qquad \nabla \cdot \mathbf{u} = 0,
$$

starting from the analytic vortex
$\mathbf{u}_0 = (\cos x \sin y \cos z,\; -\sin x \cos y \cos z,\; 0)$.
The initial vortex sheet stretches, rolls up, and cascades to small scales.

<figure class="example-figure">

![Taylor-Green initial vorticity](/jaxpi2/gallery/taylor_green.png)

<figcaption>Initial z-vorticity on the z = 0 plane.</figcaption>
</figure>

## Run

```bash
cd examples/taylor_green

# Standard time-window training
python3 main.py --config=configs/baseline.py

# Multi-stage eps-homotopy training
python3 main.py --config=configs/multi_stage.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | Single network per time window |
| `configs/multi_stage.py` | Multi-stage correction cascade per window |
| `configs/pseudo_time.py` | Adds adaptive pseudo-time stepping |

## Notes

- The flagship example for [multi-stage training](/guide/training-techniques#multi-stage-homotopy):
  each stage trains a correction network around the frozen previous stages
  ($\text{sol} = \text{prev} + \varepsilon\,\text{diff}$) with a linearized Navier–Stokes
  residual, a larger Fourier frequency, and rejection-sampled collocation points where the
  previous residual is large.
- Long cascades are fully resumable: per-window, per-stage checkpoints
  (`time_window_{w}_stage_{s}`) plus `--config.training.resume=True`.
- Enstrophy of the predicted field is logged at every window boundary as a physical sanity
  check against the known Re 1600 reference curve.
