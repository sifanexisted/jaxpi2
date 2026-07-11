# Rayleigh–Taylor

Buoyancy-driven instability in the Boussinesq approximation: a cold, dense layer sits
above a hot, light one, and the interface rolls up into the classic mushroom plume. The
system couples momentum, continuity, and temperature transport for $(u, v, p, T)$ on
$(x, y) \in [0,1] \times [0,2]$, with no-slip top/bottom walls.

## Results

<div class="result-glance">
  <span>temperature error <strong>8.3e-03</strong></span>
  <span>recipe <strong>5 time windows + adaptive pseudo-time</strong></span>
  <span>5 × 100k steps, single GPU</span>
</div>

The starkest baseline-vs-pseudo-time contrast in the suite. Trained plainly, the network
never leaves the quiescent state — the velocity errors sit at **0.99** for the entire run
while the residual loss happily decreases: a textbook spurious solution, since "nothing
moves" nearly satisfies the equations early on. With
[adaptive pseudo-time stepping](/methods/pseudo-time) and five
[time windows](/guide/training-techniques#time-window-curriculum), the instability develops
correctly and the **entire trajectory** — from the first interface perturbation through the
fully developed plume — is captured to a stitched temperature error of **8.3e-03**
(u: 0.030, v: 0.017): the mushroom cap, stem roll-ups, and side vortices are all
quantitatively right, with the error confined to the thin thermal filaments. Notably, each
window trains from a **fresh initialization** (`transfer_learning=False`): warm-starting a
window from the previous window's parameters traps the optimizer and the same run diverges
to O(1) velocity errors.

<figure class="example-figure">
  <video src="/jaxpi2/results/rayleigh_taylor_pred.mp4" autoplay loop muted playsinline></video>
  <figcaption>Reference, PINN prediction, and absolute error of the temperature field as
  the plume develops.</figcaption>
</figure>

<figure class="example-figure">

![Rayleigh-Taylor convergence](/jaxpi2/results/rayleigh_taylor_convergence.png)

<figcaption>Training losses and per-variable errors across all five windows (the sawtooth
marks each window's fresh restart).</figcaption>
</figure>

## Run

```bash
cd examples/rayleigh_taylor
python3 main.py --config=configs/pseudo_time.py \
    --config.time_range="(0.0, 9.75)" \
    --config.training.num_time_windows=5 \
    --config.training.transfer_learning=False
```

| Config | Description |
| --- | --- |
| `configs/baseline.py` | PirateNet + causal weighting |
| `configs/pseudo_time.py` | Adaptive pseudo-time stepping (showcase) |
| `configs/fixed_pseudo_time.py` | Constant pseudo-time weights |

## Notes

- Four residual components keyed by variable (`u`, `v`, `p`, `temp`), each with its own
  adaptive loss and pseudo-time weight — see the
  [variable-keyed residual convention](/guide/concepts#the-variable-keyed-residual-convention).
- Boundary conditions flow through the batch via a dedicated `BCSampler`
  (top/bottom walls at every step).
