# Burgers

Viscous Burgers equation developing a steep internal layer:

$$
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}
  = \frac{0.01}{\pi}\,\frac{\partial^2 u}{\partial x^2},
\qquad (t, x) \in [0, 1] \times [-1, 1],
$$

with $u(0, x) = -\sin(\pi x)$ and homogeneous Dirichlet boundaries. A viscous shock forms
at $x = 0$ and sharpens over time.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>4.0e-05</strong></span>
  <span>recipe <strong>baseline</strong> — architecture-insensitive</span>
  <span>100k steps, single GPU</span>
</div>

The baseline recipe resolves the shock to a relative L2 error of **4.0e-05**, with the
residual error confined to the thin viscous layer. Burgers turns out to be a *saturated*
benchmark for modern PINN training: in a parameter-matched comparison, a plain MLP, a
modified MLP, and PirateNet all land within 1% of each other, and pseudo-time stepping changes
nothing — with SOAP and grad-norm balancing in place, every reasonable configuration
reaches the same answer. Treat it as a correctness check rather than a discriminating
benchmark.

<figure class="example-figure">

![Burgers prediction vs reference](/jaxpi2/results/burgers_pred.png)

<figcaption>Reference, PINN prediction, and absolute error; the error is confined to the shock.</figcaption>
</figure>

<figure class="example-figure">

![Burgers convergence](/jaxpi2/results/burgers_convergence.png)

<figcaption>Training losses and relative L2 error of the showcase run.</figcaption>
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
