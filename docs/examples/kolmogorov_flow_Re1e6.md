# Kolmogorov Flow at Re 10⁶

The showcase problem: forced 2D turbulence at a viscosity a thousand times smaller than the
standard benchmark, trained forward in time from a DNS snapshot:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\,\mathbf{u}
  + \nabla p - \nu \Delta \mathbf{u} = \big(0.1 \sin(4\pi y),\, 0\big),
\qquad
\nabla \cdot \mathbf{u} = 0,
\qquad \nu = 10^{-6}.
$$

<figure class="example-figure">

<video src="/jaxpi2/gallery/kolmogorov_flow_Re1e6.mp4" autoplay loop muted playsinline width="420"></video>

<figcaption>DNS reference vorticity on a 2048² grid.</figcaption>
</figure>

## Run

The DNS reference data (~1.3 GB) is not tracked in git; place it under `data/` or point the
config at it (see `data/README.md`):

```bash
cd examples/kolmogorov_flow_Re1e6
python3 main.py --config=configs/baseline.py \
    --config.data_path=/path/to/kolmogorov_flow_Re1e6.npy
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting, time windows |
| `configs/pseudo_time.py` | Adds adaptive pseudo-time stepping |

## Notes

- Time-window training with resume: `--config.training.resume=True` continues from the last
  trained window after an interruption.
- Initial conditions for each window are predicted on the full 2048² grid in batches
  (`predict_in_batches`), using the schedule-free averaged parameters.
- Enstrophy at each window boundary is logged as a physical sanity check.
