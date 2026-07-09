# Gray–Scott

Reaction–diffusion pattern formation (self-replicating spots and labyrinths):

$$
\frac{\partial u}{\partial t} = \varepsilon_1 \nabla^2 u + b_1 (1 - u) - c_1 u v^2,
\qquad
\frac{\partial v}{\partial t} = \varepsilon_2 \nabla^2 v - b_2 v + c_2 u v^2 .
$$

<figure class="example-figure">
  <video src="/jaxpi2/gallery/gray_scott.mp4" autoplay loop muted playsinline></video>
  <figcaption>Reference v-field: pattern growth from a localized seed.</figcaption>
</figure>

## Run

```bash
cd examples/gray_scott
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- The patterns emerge from an unstable equilibrium — exactly the setting where PINNs are
  prone to spurious steady solutions, making this a key benchmark for pseudo-time stepping.
- Supports a fixed residual batch (`--config.training.random_sampling=False`) for
  controlled pseudo-time experiments.
