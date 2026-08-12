# Reference

JAXPI integrates training strategies from a line of work on why physics-informed neural
networks fail and how to fix them. Short summaries and pointers below; the
[Methods](/methods/piratenet) section has illustrated deep dives into each component of the
baseline (architecture, loss balancing, causal training, SOAP, pseudo-time stepping,
forward-mode residual derivatives).

## Failure modes of PINN training

**Gradient pathologies.** Multi-term PINN losses produce imbalanced gradients; boundary terms
are overwhelmed by residual terms (or vice versa), and training stalls. Grad-norm loss
balancing (the `loss_weighting` module) equalizes the per-term gradient magnitudes.
— [Wang, Teng & Perdikaris (2021), *Understanding and Mitigating Gradient Flow Pathologies in
PINNs*](https://epubs.siam.org/doi/10.1137/20M1318043)

**Spectral bias.** Coordinate MLPs learn low frequencies first and may never fit fine scales.
Random Fourier features re-shape the NTK spectrum and are the default front-end for every
architecture here.
— [Tancik et al. (2020)](https://arxiv.org/abs/2006.10739),
[Wang, Wang & Perdikaris (2021)](https://www.sciencedirect.com/science/article/abs/pii/S0045782521002759)

**NTK perspective.** The convergence of PINN training is governed by the neural tangent
kernel of the composite residual operator; ill-conditioning explains both pathologies above.
— [Wang, Yu & Perdikaris (2022)](https://www.sciencedirect.com/science/article/pii/S002199912100663X)

**Violated causality.** Minimizing the residual uniformly in time lets the network fit late
times from wrong early dynamics. Causal weighting restores the temporal ordering of
convergence.
— [Wang, Sankaran & Perdikaris (2024), *Respecting Causality for Training
PINNs*](https://www.sciencedirect.com/science/article/pii/S0045782524000690)

**Spurious solutions.** For PDEs with unstable equilibria or non-unique weak solutions, exact
residual minimizers exist that are physically wrong. Pseudo-time stepping augments the
residual with an implicit-Euler-like damping term that removes these attractors.
— [Wang, Koohy, Lu & Perdikaris (2026), *When PINNs Go Wrong: Pseudo-Time Stepping Against
Spurious Solutions*](https://arxiv.org/abs/2604.23528v1)

## Architectures

**PirateNets.** Residual adaptive networks whose blocks interpolate between identity and a
gated transformation via a learnable $\alpha$, allowing stable training of deep physics
networks.
— [Wang, Li, Chen & Perdikaris (2024), JMLR](https://arxiv.org/abs/2402.00326)

**Exact periodicity.** Hard-constraining periodic boundary conditions through cos/sin input
embeddings.
— [Dong & Ni (2021)](https://www.sciencedirect.com/science/article/abs/pii/S0021999121001376)

**Random weight factorization.** Reparameterizing weights as scale × direction accelerates
training of coordinate networks.
— [Wang et al. (2022)](https://arxiv.org/abs/2210.01274)

## Optimization

**Second-order alignment.** Quasi-Newton preconditioning (SOAP) aligns the gradients of
competing loss terms and dramatically improves PINN convergence; JAXPI ships SOAP and Muon
alongside Adam, with an optional schedule-free wrapper.
— [Wang et al. (2025), *Gradient Alignment in Physics-informed Neural Networks*, NeurIPS](https://arxiv.org/abs/2502.00604)

## Further reading

- [Wang, Sankaran, Wang & Perdikaris (2023), *An Expert's Guide to Training
  PINNs*](https://arxiv.org/abs/2308.08468) — the practical playbook most of the defaults
  in this library come from.
- [Krishnapriyan et al. (2021), *Characterizing Possible Failure Modes in
  PINNs*](https://arxiv.org/abs/2109.01050)

See [Training Techniques](/guide/training-techniques) for how each idea maps to a config
switch.
