# Ginzburg–Landau

The real Ginzburg–Landau system in 2D, written for the pair $(u, v)$ — spiral waves emerge
from the reaction terms:

$$
\frac{\partial u}{\partial t} = \varepsilon \nabla^2 u
  + k \left[ u - (u^2 + v^2)\,u - 1.5\,(u^2 + v^2)\,v \right],
$$

$$
\frac{\partial v}{\partial t} = \varepsilon \nabla^2 v
  + k \left[ v - (u^2 + v^2)\,v + 1.5\,(u^2 + v^2)\,u \right].
$$

## Results

<div class="result-glance">
  <span>relative L2 error <strong>0.011</strong></span>
  <span>recipe <strong>adaptive pseudo-time</strong></span>
  <span>100k steps, single GPU</span>
</div>

The rotating spiral is captured to a relative L2 error of **0.011** with adaptive
pseudo-time stepping — a 4× improvement over the baseline's 0.042. Ginzburg–Landau has the
classic spurious-attractor structure (the homogeneous state is a residual minimizer), and
the damping term is what keeps training locked onto the rotating solution as the wavefront
sweeps the domain. The animation below shows the prediction tracking the reference through
a full rotation; the error stays confined to the spiral arms where the field gradients are
steepest.

<figure class="example-figure">
  <video src="/jaxpi2/results/ginzburg_landau_pred.mp4" autoplay loop muted playsinline></video>
  <figcaption>Reference, PINN prediction, and absolute error of the u-field over time.</figcaption>
</figure>

<figure class="example-figure">

![Ginzburg-Landau convergence](/jaxpi2/results/ginzburg_landau_convergence.png)

<figcaption>Training losses and per-variable relative L2 errors of the showcase run.</figcaption>
</figure>

## Run

```bash
cd examples/ginzburg_landau
python3 main.py --config=configs/pseudo_time.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping (showcase) |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Time-window training with `MeshSampler` initial conditions: the IC data flows through the
  batch, so each window simply resamples from the propagated field.
- The two residuals are keyed by their variables (`u`, `v`) — see the
  [variable-keyed residual convention](/guide/concepts#the-variable-keyed-residual-convention).
