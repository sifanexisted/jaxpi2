# Kuramoto–Sivashinsky

Spatiotemporal chaos with fourth-order hyper-diffusion:

$$
\frac{\partial u}{\partial t} + \alpha\, u\,\frac{\partial u}{\partial x}
  + \beta\,\frac{\partial^2 u}{\partial x^2}
  + \gamma\,\frac{\partial^4 u}{\partial x^4} = 0,
\qquad
\alpha = \tfrac{100}{16},\;\; \beta = \tfrac{100}{16^2},\;\; \gamma = \tfrac{100}{16^4},
$$

with periodic boundaries, trained through the transition to chaos.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>4.3e-03</strong></span>
  <span>recipe <strong>adaptive pseudo-time</strong></span>
  <span>100k steps, single GPU</span>
</div>

The chaotic trajectory is tracked to a relative L2 error of **4.3e-03** — and here
pseudo-time stepping is transformative, cutting the baseline's 9.4e-02 by **22×**. As the
dynamics transition to chaos, the plain residual loss develops transient minima that pull
training off the trajectory; the damping term suppresses exactly those excursions. Two
further observations sharpen the picture: with pseudo-time active, causal weighting
becomes redundant (turning it off even edges slightly ahead at 3.6e-03), and a
parameter-matched comparison of PirateNet, ModifiedMlp, and a plain MLP shows all three
tie at the baseline level — the method, not the network, is what moves this benchmark.

<figure class="example-figure">

![KS prediction vs reference](/jaxpi2/results/ks_pred.png)

<figcaption>Reference, PINN prediction, and absolute error through the transition to chaos.</figcaption>
</figure>

<figure class="example-figure">

![KS convergence](/jaxpi2/results/ks_convergence.png)

<figcaption>Training losses and relative L2 error of the showcase (pseudo-time) run.</figcaption>
</figure>

## Run

```bash
cd examples/ks
python3 main.py --config=configs/pseudo_time.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping (showcase) |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Longer horizons use [time-window training](/guide/training-techniques#time-window-curriculum)
  (`--config.training.num_time_windows=4`), propagating the network's own prediction as the
  next initial condition.
- Fourth derivatives via Taylor-mode AD (`jax.experimental.jet`).
