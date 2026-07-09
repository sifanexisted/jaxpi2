# Burgers

Viscous Burgers equation developing a steep internal layer:

$$
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}
  = \frac{0.01}{\pi}\,\frac{\partial^2 u}{\partial x^2},
\qquad (t, x) \in [0, 1] \times [-1, 1],
$$

with $u(0, x) = -\sin(\pi x)$ and homogeneous Dirichlet boundaries.

<figure class="example-figure">

![Burgers space-time solution](/jaxpi2/gallery/burgers.png)

<figcaption>Reference solution u(t, x); a viscous shock forms at x = 0.</figcaption>
</figure>

## Run

```bash
cd examples/burgers
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |
