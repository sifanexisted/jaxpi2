# Gray–Scott

Reaction–diffusion pattern formation (self-replicating spots and labyrinths):

$$
\frac{\partial u}{\partial t} = \varepsilon_1 \nabla^2 u + b_1 (1 - u) - c_1 u v^2,
\qquad
\frac{\partial v}{\partial t} = \varepsilon_2 \nabla^2 v - b_2 v + c_2 u v^2 .
$$

The patterns grow from a localized seed on top of an unstable homogeneous equilibrium —
exactly the setting where PINNs are prone to spurious steady solutions.

## Results

<div class="result-glance">
  <span>relative L2 error <strong>7.0e-03</strong></span>
  <span>recipe <strong>4 time windows + adaptive pseudo-time</strong></span>
  <span>4 × 100k steps, single GPU</span>
</div>

This is the suite's most dramatic rescue. Trained naively over the whole time range, the
network converges to the spurious homogeneous state and never grows the pattern — a
relative L2 error of **0.90** despite a small training loss. Splitting the horizon into
four [time windows](/guide/training-techniques#time-window-curriculum) and enabling
[adaptive pseudo-time stepping](/methods/pseudo-time) recovers the full pattern-formation
dynamics at **7.0e-03** over the stitched trajectory — a ~130× improvement, with each
window trained from a **fresh initialization** (`transfer_learning=False`). Pseudo-time
alone (no windows) already recovers to 0.03, confirming the spurious-equilibrium
diagnosis; the windowed curriculum then removes the remaining long-horizon error.

<figure class="example-figure">
  <video src="/jaxpi2/results/gray_scott_pred.mp4" autoplay loop muted playsinline></video>
  <figcaption>Reference, PINN prediction, and absolute error of the v-field: pattern growth from a localized seed.</figcaption>
</figure>

<figure class="example-figure">

![Gray-Scott convergence](/jaxpi2/results/gray_scott_convergence.png)

<figcaption>Training losses and per-variable errors across all four windows (the sawtooth
marks each window's fresh restart).</figcaption>
</figure>

## Run

```bash
cd examples/gray_scott
python3 main.py --config=configs/pseudo_time.py \
    --config.time_range="(0.0, 1.0)" \
    --config.training.num_time_windows=4 \
    --config.training.transfer_learning=False
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping (showcase, with 4 windows) |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- A key benchmark for [pseudo-time stepping](/methods/pseudo-time): the unstable homogeneous
  equilibrium is a perfect residual minimizer that plain training happily finds.
- Supports a fixed residual batch (`--config.training.random_sampling=False`) for
  controlled pseudo-time experiments.
