# Wave

Second-order wave equation with a two-mode standing-wave solution:

$$
\frac{\partial^2 u}{\partial t^2} = c^2\,\frac{\partial^2 u}{\partial x^2},
\qquad c = 2,\quad (t, x) \in [0, 1] \times [0, 1],
$$

with $u(0,x) = \sin(\pi x) + a \sin(2\pi x)$, $u_t(0,x) = 0$, and homogeneous Dirichlet
boundaries. Both the displacement and velocity initial conditions enter the loss.

<figure class="example-figure">

![Wave space-time solution](/jaxpi2/gallery/wave.png)

<figcaption>Reference solution u(t, x).</figcaption>
</figure>

## Run

```bash
cd examples/wave
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |
