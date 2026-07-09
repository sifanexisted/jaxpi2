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

<figure class="example-figure">
  <video src="/jaxpi2/gallery/ginzburg_landau.mp4" autoplay loop muted playsinline></video>
  <figcaption>Reference u-field: rotating spiral waves.</figcaption>
</figure>

## Run

```bash
cd examples/ginzburg_landau
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Time-window training with `MeshSampler` initial conditions: the IC data flows through the
  batch, so each window simply resamples from the propagated field.
- The two residuals are returned as `{"ru", "rv"}` for per-component adaptive weighting.
