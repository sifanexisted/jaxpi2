# Inviscid Burgers

The vanishing-viscosity limit — a genuine shock forms in finite time and the PDE residual
becomes ill-defined at the discontinuity:

$$
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = 0 .
$$

<figure class="example-figure">

![Inviscid Burgers space-time solution](/jaxpi2/gallery/inviscid_burgers.png)

<figcaption>Reference solution u(t, x) with a moving shock.</figcaption>
</figure>

## Run

```bash
cd examples/inviscid_burgers
python3 main.py --config=configs/baseline.py
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Without regularization PINNs converge to spurious weak solutions here; pseudo-time
  stepping biases training toward the entropy solution
  (see [Training Techniques](/guide/training-techniques)).
- Supports a fixed collocation batch (`--config.training.random_sampling=False`) to isolate
  the effect of the pseudo-time weights from resampling noise.
