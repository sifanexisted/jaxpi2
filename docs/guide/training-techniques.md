# Training Techniques

Hard PDEs break naive PINN training in reproducible ways: stiff dynamics get skipped, unstable
equilibria attract the optimizer, shocks produce spurious weak solutions. JAXPI packages the
counter-measures as config switches that compose freely.

## Grad-norm loss balancing

**Problem.** Multi-term losses ($\mathcal{L}_{ic} + \mathcal{L}_{bc} + \mathcal{L}_{res}$)
have wildly different gradient scales; one term dominates and the others stall.

**Fix.** With `loss_weighting.strategy = "dynamic"`, weights are periodically set to

$$
\lambda_i \leftarrow \frac{\overline{\lVert \nabla_\theta \mathcal{L} \rVert}}{\lVert \nabla_\theta \mathcal{L}_i \rVert},
$$

then smoothed with momentum, so all terms contribute equally-sized gradients.

```python
loss_weighting.strategy = "dynamic"
loss_weighting.update_schedule = {"start": 100, "every": 1000}
loss_weighting.momentum = 0.9
```

## Causal training

**Problem.** For time-dependent PDEs, the residual can be minimized at late times before the
early dynamics are resolved — the network "predicts the future" from wrong initial data and
converges to garbage (classic on Allen–Cahn and fast advection).

**Fix.** Sort the collocation batch by time, split it into chunks, and gate each chunk by the
cumulative residual of everything **earlier**:

$$
\mathcal{L}_{res} = \frac{1}{K} \sum_{k=1}^{K} \gamma_k\, \mathcal{L}_k,
\qquad
\gamma_k = \exp\Big( -\tau \sum_{j < k} \mathcal{L}_j \Big).
$$

Late chunks only receive weight once earlier chunks have converged.

```python
causal.enabled = True
causal.num_chunks = 32
causal.tol = 1.0
```

Under multi-GPU sharding, chunk losses are gathered across devices in global time order, so
causality is exact for any device count.

## Pseudo-time stepping

**Problem.** PDEs with unstable equilibria (Gray–Scott, Ginzburg–Landau, high-Re cavity flow)
admit spurious steady solutions that are perfect residual minimizers — the optimizer happily
finds them.

**Fix.** Augment the residual with a damping term toward the previous iterate, which mimics an
implicit pseudo-time discretization and destabilizes the spurious fixed points:

$$
\tilde{r}(\theta) = r(\theta) + w \, \big( u_\theta - u_{\theta_{prev}} \big).
$$

The per-component weights $w$ can be constant (`strategy = "constant"`) or adapted from the
ratio of residual change to solution change (`strategy = "dynamic"`), with an optional cosine
shrink as the residual converges:

```python
pseudo_time.enabled = True
pseudo_time.strategy = "dynamic"
pseudo_time.pts_weights = {"ru": 1.0, "rv": 1.0, "rc": 1.0}
pseudo_time.shrink.enabled = True
```

## Time-window curriculum

**Problem.** Chaotic trajectories (Kuramoto–Sivashinsky, Kolmogorov flow) cannot be fit
globally — small early errors amplify exponentially.

**Fix.** Train on consecutive windows $[k \Delta T, (k+1) \Delta T]$, propagating the network's
own prediction as the next initial condition, with parameter transfer between windows. This is
`train_time_windows` (see [The Trainer](/guide/trainer)):

```python
training.num_time_windows = 10
training.time_window_size = 0.2   # examples that define dT explicitly
training.transfer_learning = True
training.resume = True            # continue an interrupted cascade
```

## Multi-stage homotopy

**Problem.** Even a well-trained network plateaus at a finite residual; multi-scale flows
(Taylor–Green at Re 1600) need more accuracy than one network can deliver.

**Fix.** Freeze the trained solution and train a *correction* network on the linearized
equations around it,

$$
u = u_{prev} + \varepsilon\, u_{diff},
\qquad
f(u_{diff}) + \frac{r(u_{prev})}{\varepsilon} = 0,
$$

with a higher Fourier frequency (corrections live at finer scales) and extra collocation
points rejection-sampled where the previous residual is large. Stages compose:
each new stage corrects the sum of all previous ones. See
`examples/taylor_green/train_multi_stage.py`.

## Composition cheat sheet

| Problem symptom | Reach for |
| --- | --- |
| One loss term dominates | grad-norm balancing |
| Solution ignores initial condition | causal training |
| Converges to a trivial/steady solution | pseudo-time stepping |
| Chaotic / long-time dynamics | time windows |
| Residual plateaus above target accuracy | multi-stage correction |
