# Advection

1D linear transport of a periodic profile at high speed — the classic sanity check for causal
PINN training, since the solution must propagate strictly forward in time:

$$
\frac{\partial u}{\partial t} + c\,\frac{\partial u}{\partial x} = 0,
\qquad c = 30,\quad (t, x) \in [0, 2] \times [0, 2\pi],
$$

with periodic boundary conditions and $u(0, x) = \sin x$. The exact solution
$u(t,x) = \sin(x - ct)$ is used as the reference.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>4.3e-02</strong></span>
  <span>recipe <strong>baseline + causal tol 0.01</strong></span>
  <span>100k steps, single GPU</span>
</div>

Fast transport turns out to hinge on one hyperparameter: the **causal tolerance**. With
the default `tol = 1.0`, the large residuals of high-speed advection drive every causal
gate to zero — only the earliest sliver of time ever trains, the initial condition is fit
perfectly, and the rest of the domain is garbage (relative L2 error stuck at ~0.8-0.9
across every architecture and embedding we tried). Relaxing the tolerance to
`tol = 0.01` (now the config default, with 32 chunks) lets the causal front actually
sweep the domain and the same network reaches **4.3e-02**, with the error growing gently
along the transport direction. A cautionary counterexample: increasing the Fourier
embedding scale to "help" the fast dynamics made things *worse* (0.38) — the fix was in
the training schedule, not the representation.

<figure class="example-figure">

![Advection prediction vs reference](/jaxpi2/results/advection_pred.png)

<figcaption>Reference, PINN prediction, and absolute error; the transported profile stays
phase-locked across the full horizon.</figcaption>
</figure>

<figure class="example-figure">

![Advection convergence](/jaxpi2/results/advection_convergence.png)

<figcaption>Training losses and relative L2 error of the showcase run.</figcaption>
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
- The advection speed is configurable: `--config.problem.c=...`.
