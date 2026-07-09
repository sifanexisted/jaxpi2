# Korteweg–de Vries

Soliton dynamics with third-order dispersion:

$$
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}
  + \delta^2\,\frac{\partial^3 u}{\partial x^3} = 0,
\qquad \delta = 0.022,
$$

with periodic boundaries and $u(0, x) = \cos(\pi x)$. The initial profile steepens and breaks
into a train of interacting solitons.

<figure class="example-figure">

![KdV space-time solution](/jaxpi2/gallery/kdv.png)

<figcaption>Reference solution u(t, x); soliton fission and near-recurrence.</figcaption>
</figure>

## Run

```bash
cd examples/kdv
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Third derivatives are computed with Taylor-mode automatic differentiation
  (`jax.experimental.jet`), which is much cheaper than nesting `grad` three times.
