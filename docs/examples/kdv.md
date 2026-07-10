# Korteweg–de Vries

Soliton dynamics with third-order dispersion:

$$
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}
  + \delta^2\,\frac{\partial^3 u}{\partial x^3} = 0,
\qquad \delta = 0.022,
$$

with periodic boundaries and $u(0, x) = \cos(\pi x)$. The initial profile steepens and breaks
into a train of interacting solitons.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>1.5e-04</strong></span>
  <span>recipe <strong>adaptive pseudo-time</strong></span>
  <span>50k steps, single GPU</span>
</div>

Adaptive pseudo-time stepping gives the best result on KdV: **1.5e-04** relative L2 error,
about 2× better than the plain baseline (2.9e-04). The soliton train's fine dispersive
structure is where the damping term helps — it suppresses the transient oscillations the
optimizer otherwise chases during soliton fission. This example is also our cleanest
optimizer comparison: swapping SOAP for Adam at identical settings costs 8× (2.2e-03) —
see the [SOAP deep-dive](/methods/soap).

<figure class="example-figure">

![KdV prediction vs reference](/jaxpi2/results/kdv_pred.png)

<figcaption>Reference, PINN prediction, and absolute error across soliton fission and interaction.</figcaption>
</figure>

<figure class="example-figure">

![KdV convergence](/jaxpi2/results/kdv_convergence.png)

<figcaption>Training losses and relative L2 error of the showcase (pseudo-time) run.</figcaption>
</figure>

## Run

```bash
cd examples/kdv
python3 main.py --config=configs/pseudo_time.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping (showcase) |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Third derivatives are computed with Taylor-mode automatic differentiation
  (`jax.experimental.jet`), which is much cheaper than nesting `grad` three times.
