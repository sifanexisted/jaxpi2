# Wave

Second-order wave equation with a two-mode standing-wave solution:

$$
\frac{\partial^2 u}{\partial t^2} = c^2\,\frac{\partial^2 u}{\partial x^2},
\qquad c = 2,\quad (t, x) \in [0, 1] \times [0, 1],
$$

with $u(0,x) = \sin(\pi x) + a \sin(2\pi x)$, $u_t(0,x) = 0$, and homogeneous Dirichlet
boundaries. Both the displacement and velocity initial conditions enter the loss.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>1.2e-05</strong></span>
  <span>recipe <strong>baseline</strong> (dynamic grad-norm weighting essential)</span>
  <span>100k steps, single GPU</span>
</div>

The baseline recipe reproduces the standing wave to a relative L2 error of **1.2e-05**.
What makes this example instructive is *why*: the wave equation's three loss terms
(displacement IC, velocity IC, residual) have wildly different gradient scales, and our
experiments show that freezing the loss weights at
constant values degrades the error by more than three orders of magnitude (to 2.6e-02) —
the largest single-ingredient effect we measured anywhere in the suite. Grad-norm
balancing, not the architecture or the optimizer, is what carries this problem.

<figure class="example-figure">

![Wave prediction vs reference](/jaxpi2/results/wave_pred.png)

<figcaption>Reference, PINN prediction, and absolute error.</figcaption>
</figure>

<figure class="example-figure">

![Wave convergence](/jaxpi2/results/wave_convergence.png)

<figcaption>Training losses and relative L2 error of the showcase run.</figcaption>
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

## Notes

- See the [loss-balancing deep-dive](/methods/loss-balancing) for the grad-norm scheme this
  example depends on.
