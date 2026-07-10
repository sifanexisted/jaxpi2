# Inviscid Burgers

The vanishing-viscosity limit — a genuine shock forms in finite time and the PDE residual
becomes ill-defined at the discontinuity:

$$
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = 0 .
$$

Without regularization, PINNs are known to converge to *spurious weak solutions* here:
profiles that satisfy the residual almost everywhere but place the shock incorrectly.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>0.127</strong></span>
  <span>recipe <strong>adaptive pseudo-time</strong></span>
  <span>100k steps, single GPU</span>
</div>

Adaptive pseudo-time stepping reaches a relative L2 error of **0.127**, versus 0.214 for
the baseline and 0.166 for constant pseudo-time weights — consistent with the method's
core claim that the damping term steers training away from spurious weak solutions and
toward the entropy solution. The remaining error is dominated by the immediate
neighborhood of the discontinuity, where any pointwise residual is ill-defined; away from
the shock the profile is essentially exact. This is the paper-motivating example for
[pseudo-time stepping](/methods/pseudo-time), with the adaptive step size earning its keep
over the fixed one.

<figure class="example-figure">

![Inviscid Burgers prediction vs reference](/jaxpi2/results/inviscid_burgers_pred.png)

<figcaption>Reference, PINN prediction, and absolute error; the error concentrates at the moving shock.</figcaption>
</figure>

<figure class="example-figure">

![Inviscid Burgers convergence](/jaxpi2/results/inviscid_burgers_convergence.png)

<figcaption>Training losses and relative L2 error of the showcase (adaptive pseudo-time) run.</figcaption>
</figure>

## Run

```bash
cd examples/inviscid_burgers
python3 main.py --config=configs/pseudo_time.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping (showcase) |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Supports a fixed collocation batch (`--config.training.random_sampling=False`) to isolate
  the effect of the pseudo-time weights from resampling noise.
