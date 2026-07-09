# Causal Training

::: info TL;DR
Time-dependent PDEs must be solved forward in time, but the plain PINN loss lets the network
fit late times before early ones — and converge to garbage. Causal weighting gates each time
slice by the cumulative residual of everything earlier. Based on
[Wang, Sankaran & Perdikaris, *Respecting Causality for Training PINNs*, CMAME
2024](https://doi.org/10.1016/j.cma.2024.116813).
:::

## The problem: PINNs violate causality

Minimizing the residual uniformly over $[0, T]$ contains no notion of temporal ordering. The
paper shows that gradient descent on this objective implicitly minimizes *late-time*
residuals first for many PDEs — the network fits the future from wrong initial data, then
cannot recover. This is the dominant failure mode for stiff and chaotic dynamics
(Allen–Cahn, Kuramoto–Sivashinsky, turbulent flows).

## The fix: gate the residual by earlier convergence

Partition the temporal domain into $N_t$ slices and write the weighted residual loss
(paper, Eqs. 3.1–3.3):

$$
\mathcal{L}_r(\theta) = \frac{1}{N_t} \sum_{i=1}^{N_t} w_i\, \mathcal{L}_r(t_i, \theta),
\qquad
w_i = \exp\left( -\varepsilon \sum_{k=1}^{i-1} \mathcal{L}_r(t_k, \theta) \right),
$$

with $w_1 = 1$. The gates $w_i$ are treated as constants (`stop_gradient`). Slice $i$
receives significant weight only once **all earlier** residuals are small — training sweeps
through time automatically:

![Causal gates during training](/jaxpi2/methods/causal_weights.png)

The causality parameter $\varepsilon$ controls the steepness of the gate. A useful
convergence diagnostic falls out for free: training is done when $\min_i w_i \approx 1$
(JAXPI logs this as `cas_weight`).

## In JAXPI

`ForwardIVP` implements the scheme by time-sorting each collocation batch
(`UniformSampler` does this by default) and splitting it into `num_chunks` slices:

```python
causal.enabled = True
causal.num_chunks = 32   # N_t
causal.tol = 1.0         # epsilon
```

Under multi-GPU sharding, per-chunk losses are `all_gather`-ed across devices in global time
order before the gates are computed, so causality is exact for any device count — this
equivalence is covered by the test suite.

## Where it's used

Enabled in most time-dependent baselines; the cleanest demonstrations are
[Allen–Cahn](/examples/allen_cahn) (without it, the network collapses to the trivial
equilibrium) and [Advection](/examples/advection) at $c = 50$.
