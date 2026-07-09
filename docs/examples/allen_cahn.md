# Allen–Cahn

Phase separation with a stiff double-well reaction term:

$$
\frac{\partial u}{\partial t} - 10^{-4}\,\frac{\partial^2 u}{\partial x^2} + 5u^3 - 5u = 0,
\qquad (t, x) \in [0, 1] \times [-1, 1],
$$

with periodic boundary conditions and $u(0, x) = x^2 \cos(\pi x)$.

<figure class="example-figure">

![Allen-Cahn space-time solution](/jaxpi2/gallery/allen_cahn.png)

<figcaption>Reference solution u(t, x); sharp transition layers form and persist.</figcaption>
</figure>

## Run

```bash
cd examples/allen_cahn
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- A well-known PINN failure mode: without causal weighting the network converges to the
  unstable trivial equilibrium. This example is the standard testbed for the causal loss.
