# Rayleigh–Taylor

Buoyancy-driven mixing of a heavy fluid over a light one, modeled with the Boussinesq
approximation — incompressible Navier–Stokes coupled to a temperature field:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\,\mathbf{u}
  + \nabla p - \alpha_1 \nabla^2 \mathbf{u} = \alpha_2\, \theta\, \mathbf{e}_y,
\qquad \nabla \cdot \mathbf{u} = 0,
$$

$$
\frac{\partial \theta}{\partial t} + (\mathbf{u} \cdot \nabla)\,\theta
  = \alpha_4 \nabla^2 \theta .
$$

<figure class="example-figure">
  <video src="/jaxpi2/gallery/rayleigh_taylor.mp4" autoplay loop muted playsinline></video>
  <figcaption>Reference temperature field: mushroom plumes of the instability.</figcaption>
</figure>

## Run

```bash
cd examples/rayleigh_taylor
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Four residual components keyed by variable (`u`, `v`, `p`, `temp`: momentum, continuity, energy), each
  with its own adaptive weight.
- Uses a dedicated boundary-condition sampler for the walls in addition to IC and residual
  samplers — a template for problems with several point sets per batch.
