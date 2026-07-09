# Advection

1D linear transport of a periodic profile at high speed — the classic sanity check for causal
PINN training, since the solution must propagate strictly forward in time:

$$
\frac{\partial u}{\partial t} + c\,\frac{\partial u}{\partial x} = 0,
\qquad c = 50,\quad (t, x) \in [0, 2] \times [0, 2\pi],
$$

with periodic boundary conditions and $u(0, x) = \sin x$. The exact solution
$u(t,x) = \sin(x - ct)$ is used as the reference.

<figure class="example-figure">

![Advection space-time solution](/jaxpi2/gallery/advection.png)

<figcaption>Reference solution u(t, x); time runs left to right.</figcaption>
</figure>

## Run

```bash
cd examples/advection
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/plain.py` | Minimal MLP + Adam, no bells and whistles |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Exact periodicity is enforced through `PeriodEmbs` on the spatial axis, so no boundary loss is needed.
- The high advection speed makes plain PINN training collapse to trivial solutions; causal weighting fixes this (see [Training Techniques](/guide/training-techniques)).
