# Kolmogorov Flow

Two-dimensional forced turbulence on a periodic domain — incompressible Navier–Stokes with a
sinusoidal body force:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\,\mathbf{u}
  + \nabla p - \nu \nabla^2 \mathbf{u} = A \sin(k \pi y)\, \mathbf{e}_x,
\qquad \nabla \cdot \mathbf{u} = 0 .
$$

The repository ships this problem at two very different Reynolds numbers:

| Example | Re | Forcing | Reference data |
| --- | --- | --- | --- |
| `examples/kolmogorov_flow` | $2 \times 10^3$ | $2\sin(4\pi y)$ | included (`data/kolmogorov_flow.npy`) |
| `examples/kolmogorov_flow_Re1e6` | $10^6$ | $0.1\sin(4\pi y)$ | external DNS snapshot (see its `data/README.md`) |

<figure class="example-figure">

![Kolmogorov flow vorticity at Re 1e6](/jaxpi2/gallery/kolmogorov_flow_Re1e6.png)

<figcaption>DNS vorticity at Re 10⁶ — fine filamentation over six orders of magnitude in scale.</figcaption>
</figure>

<figure class="example-figure">
  <video src="/jaxpi2/gallery/kolmogorov_flow.mp4" autoplay loop muted playsinline></video>
  <figcaption>Reference vorticity evolution at Re 2000.</figcaption>
</figure>

## Run

```bash
# Moderate Reynolds number (reference data included)
cd examples/kolmogorov_flow
python3 main.py --config=configs/baseline.py

# Re = 1e6 (point data_path at the DNS snapshot)
cd examples/kolmogorov_flow_Re1e6
python3 main.py --config=configs/baseline.py \
    --config.data_path=/path/to/kolmogorov_flow_Re1e6.npy
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting + time windows |
| `configs/pseudo_time.py` | Adds adaptive pseudo-time stepping |

## Notes

- Both variants train forward in time with [time windows](/guide/training-techniques#time-window-curriculum):
  the network's own prediction at the end of each window becomes the next initial condition,
  and `--config.training.resume=True` continues an interrupted cascade.
- Vorticity is never an output: $\omega = v_x - u_y$ is obtained by differentiating the
  network, so the momentum residuals `{"ru", "rv"}` and continuity `{"rc"}` each carry their
  own adaptive weight.
- The Re 10⁶ case uses schedule-free SOAP and evaluates initial conditions with the averaged
  parameters (`get_eval_params`).
