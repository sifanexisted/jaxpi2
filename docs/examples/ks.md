# Kuramoto–Sivashinsky

Spatiotemporal chaos with fourth-order hyper-diffusion — trained forward in time over
windows because a single global fit cannot track chaotic trajectories:

$$
\frac{\partial u}{\partial t} + \alpha\, u\,\frac{\partial u}{\partial x}
  + \beta\,\frac{\partial^2 u}{\partial x^2}
  + \gamma\,\frac{\partial^4 u}{\partial x^4} = 0,
\qquad
\alpha = \tfrac{100}{16},\;\; \beta = \tfrac{100}{16^2},\;\; \gamma = \tfrac{100}{16^4},
$$

with periodic boundaries.

<figure class="example-figure">

![Kuramoto-Sivashinsky space-time solution](/jaxpi2/gallery/ks.png)

<figcaption>Reference solution u(t, x): transition to chaos.</figcaption>
</figure>

## Run

```bash
cd examples/ks
python3 main.py --config=configs/baseline.py \
    --config.training.num_time_windows=4
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Uses [time-window training](/guide/training-techniques#time-window-curriculum): the network
  is retrained on consecutive windows, propagating its own prediction as the next initial
  condition (with transfer learning between windows).
- Fourth derivatives via Taylor-mode AD (`jax.experimental.jet`).
