# SOAP & Gradient Alignment

::: info TL;DR
Beyond magnitude imbalance, PINN loss terms pull in *conflicting directions*, forcing
first-order optimizers into zigzag trajectories. Quasi-second-order optimizers — SOAP in
particular — precondition these conflicts away and deliver 2–10× accuracy gains. Based on
[Wang, Bhartari, Li & Perdikaris, NeurIPS 2025](https://arxiv.org/abs/2502.00604) and
[Vyas et al., *SOAP*](https://arxiv.org/abs/2409.11321).
:::

## The problem: directional gradient conflicts

PINN training suffers two distinct gradient pathologies. Magnitude imbalance is handled by
[loss balancing](/methods/loss-balancing). The second is directional: gradients of competing
terms (e.g. no-slip boundary conditions vs. bulk momentum conservation) point in conflicting
directions, and a first-order optimizer must follow their average — an inefficient zigzag.

The paper quantifies this with an **alignment score** that generalizes cosine similarity to
$n$ gradients (score 1 = perfectly aligned, 0 = orthogonal, −1 = opposed), measured both
across loss terms within a step and between consecutive steps. First-order optimizers
oscillate near or below zero early in training; quasi-second-order ones stay positive.

![Zigzag vs preconditioned trajectories](/jaxpi2/methods/soap_alignment.png)

## The fix: precondition with SOAP

[SOAP](https://arxiv.org/abs/2409.11321) runs Adam in Shampoo's eigenbasis: it maintains a
Kronecker-factored preconditioner per weight matrix, rotates gradients into its eigenbasis,
and applies Adam there. The NeurIPS paper connects this to Newton's method and shows that,
among Adam / Muon / Kron / SOAP, **SOAP achieves the highest alignment scores** — and with
it, state-of-the-art PINN results on 10 benchmarks, including the first PINN solutions of
turbulent flows at $\mathrm{Re}$ up to $10^4$.

Intuition: near the optimum, a Newton-like preconditioner rescales the loss landscape to be
locally isotropic, so gradients of different terms stop fighting over direction and
consecutive steps stop reversing each other.

## In JAXPI

SOAP is the default optimizer for the hard benchmarks; Adam and Muon are available as
drop-ins:

```python
optim.optimizer = "soap"     # or "adam", "muon"
optim.learning_rate = 1e-3
optim.schedule_free = True   # optional schedule-free wrapper
```

With `schedule_free = True`, evaluation should use the averaged parameters — see
[Checkpointing & Resume](/guide/checkpointing#schedule-free-evaluation).

## Where it's used

Every flow benchmark ([Kolmogorov flow](/examples/kolmogorov_flow),
[Taylor–Green](/examples/taylor_green), [lid-driven cavity](/examples/lid_driven_cavity))
trains with SOAP; switch any example to Adam with `--config.optim.optimizer=adam` to see the
difference for yourself.
