# Self-Adaptive Loss Balancing

::: info TL;DR
PINN losses have several terms with wildly different gradient scales. JAXPI periodically
rescales each term so that all weighted gradients have equal norm — no manual weight tuning.
Based on [Wang, Teng & Perdikaris (2021)](https://epubs.siam.org/doi/10.1137/20M1318043) and
Algorithm 1 of the [Expert's Guide (2023)](https://arxiv.org/abs/2308.08468).
:::

## The problem: one gradient drowns the others

A PINN minimizes a composite objective,

$$
\mathcal{L}(\theta) = \lambda_{ic}\,\mathcal{L}_{ic}(\theta)
  + \lambda_{bc}\,\mathcal{L}_{bc}(\theta)
  + \lambda_{r}\,\mathcal{L}_{r}(\theta),
$$

where the residual term involves PDE derivatives of the network and typically produces
gradients orders of magnitude larger (or smaller) than the data-fit terms. With uniform
weights, gradient descent effectively optimizes only the dominant term — the classic
"gradient pathology" of PINNs.

![Gradient norms before and after balancing](/jaxpi2/methods/loss_balancing.svg)

## The fix: equalize gradient norms

Every $f$ steps, compute the candidate weights (Expert's Guide, Eqs. 2.12–2.14):

$$
\hat{\lambda}_i \;=\;
\frac{\sum_{j} \big\lVert \nabla_\theta \mathcal{L}_j(\theta) \big\rVert}
     {\big\lVert \nabla_\theta \mathcal{L}_i(\theta) \big\rVert},
\qquad i \in \{ic,\, bc,\, r\},
$$

which guarantees that all weighted gradients have equal norm:

$$
\lVert \hat{\lambda}_{ic} \nabla_\theta \mathcal{L}_{ic} \rVert
= \lVert \hat{\lambda}_{bc} \nabla_\theta \mathcal{L}_{bc} \rVert
= \lVert \hat{\lambda}_{r} \nabla_\theta \mathcal{L}_{r} \rVert
= \textstyle\sum_j \lVert \nabla_\theta \mathcal{L}_j \rVert .
$$

The actual weights are then smoothed with a moving average (Eq. 2.15):

$$
\lambda_{\text{new}} = \alpha\, \lambda_{\text{old}} + (1 - \alpha)\, \hat{\lambda}_{\text{new}},
\qquad \alpha = 0.9 .
$$

The weights are treated as constants (no gradient flows through them), and the recommended
update frequency is $f \approx 1000$ steps — the scheme costs one extra set of per-term
gradients only when it fires.

## In JAXPI

`compute_loss_weights` implements the scheme with one harmless deviation: it normalizes by
the **mean** of the gradient norms rather than the sum, which rescales all weights by the
same constant $1/n$ (equivalent to a global learning-rate factor):

```python
loss_weighting.strategy = "dynamic"
loss_weighting.loss_weights = {"ics": 1.0, "res": 1.0}   # initial values
loss_weighting.update_schedule = {"start": 100, "every": 1000}   # f
loss_weighting.momentum = 0.9                                    # alpha
```

Every loss term returned by your model's `losses()` dict gets its own weight — including
each named residual loss (`"u_res"`, `"v_res"`, `"p_res"`, ...). Under multi-GPU sharding the
per-term gradients are averaged across devices before the norms are taken, so the weights
are identical on every device.

## Where it's used

Enabled in the baseline config of **every** example — it is the single most broadly useful
trick in the toolbox. See it interact with the other techniques in
[Training Techniques](/guide/training-techniques).
