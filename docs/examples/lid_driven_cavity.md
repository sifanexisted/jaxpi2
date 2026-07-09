# Lid-driven Cavity

Steady incompressible flow in a square cavity driven by a moving lid — a boundary-value
problem (no time variable):

$$
(\mathbf{u} \cdot \nabla)\,\mathbf{u} + \nabla p - \nu \nabla^2 \mathbf{u} = 0,
\qquad \nabla \cdot \mathbf{u} = 0 \quad \text{in } [0,1]^2,
$$

with $u = 1$ on the lid and no-slip walls elsewhere. The benchmark runs up to
$\mathrm{Re} = 5000$, where a strong primary vortex and corner eddies coexist.

<figure class="example-figure">

![Lid-driven cavity velocity magnitude](/jaxpi2/gallery/lid_driven_cavity.png)

<figcaption>Reference velocity magnitude at Re 5000.</figcaption>
</figure>

## Run

```bash
cd examples/lid_driven_cavity
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | Modified MLP baseline |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Steady problems subclass `ForwardBVP` — same trainer, no causal weighting.
- At high Re, plain training stalls in a spurious low-circulation solution; pseudo-time
  stepping acts as an artificial-time continuation toward the steady state.
- Reference data ships for Re ∈ {100, 400, 1000, 1600, 3200, 5000}
  (`--config.Re=...`).
