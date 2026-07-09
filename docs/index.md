---
layout: home

hero:
  name: JAXPI
  text: Physics-informed neural networks at scale
  tagline: A lean JAX library for training PINNs on hard PDEs — multi-GPU sharding, curriculum training strategies, and 16 reproducible benchmarks.
  image:
    src: /gallery/kolmogorov_flow.png
    alt: Kolmogorov flow vorticity
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: Examples Gallery
      link: /examples/
    - theme: alt
      text: GitHub
      link: https://github.com/sifanexisted/jaxpi2

features:
  - icon: ⚡
    title: Multi-GPU by default
    details: Data-parallel training via jax.shard_map with exact gradient averaging — the same script runs on 1 or N GPUs, bit-for-bit consistent.
  - icon: 🌀
    title: Training algorithms that work
    details: Grad-norm loss balancing, causal training, pseudo-time stepping, time-window curricula, and multi-stage homotopy — all toggled from the config.
  - icon: 🏴‍☠️
    title: Modern architectures
    details: PirateNets with residual adaptive blocks, modified MLPs, random Fourier features, and exact periodic embeddings.
  - icon: 🧪
    title: 16 benchmark examples
    details: From 1D advection to 3D Taylor–Green and Kolmogorov flow at Re 10⁶ — each a self-contained script with configs and an evaluation notebook.
  - icon: 🧰
    title: One-line trainer
    details: train(config, model, sampler) handles stepping, adaptive weights, logging, checkpointing, and resume. Time windows are one call more.
  - icon: ✅
    title: Tested core
    details: A CPU-runnable test suite covers the sharded training step, causal weighting, checkpointing, and the trainer — including multi-device equivalence.
---

## At a glance

Define your PDE residual, pick a config, and train:

```python
from jaxpi.models import ForwardIVP, create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

class Burgers(ForwardIVP):
    def r_net(self, params, t, x):
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_x = grad(self.neural_net, argnums=2)(params, t, x)
        u_xx = grad(grad(self.neural_net, argnums=2), argnums=2)(params, t, x)
        return u_t + u * u_x - 0.01 / jnp.pi * u_xx

model = create_model(config, Burgers, u0=u0, t_star=t_star, x_star=x_star)
train(config, model, UniformSampler(dom, config.training.batch_size))
```

## Gallery

<div class="teaser-strip">
  <a href="/jaxpi2/examples/kolmogorov_flow"><img src="/jaxpi2/gallery/kolmogorov_flow_Re1e6.png" alt="Kolmogorov flow at Re 1e6"></a>
  <a href="/jaxpi2/examples/rayleigh_taylor"><img src="/jaxpi2/gallery/rayleigh_taylor.png" alt="Rayleigh-Taylor instability"></a>
  <a href="/jaxpi2/examples/gray_scott"><img src="/jaxpi2/gallery/gray_scott.png" alt="Gray-Scott patterns"></a>
  <a href="/jaxpi2/examples/ks"><img src="/jaxpi2/gallery/ks.png" alt="Kuramoto-Sivashinsky chaos"></a>
  <a href="/jaxpi2/examples/taylor_green"><img src="/jaxpi2/gallery/taylor_green.png" alt="Taylor-Green vortex"></a>
  <a href="/jaxpi2/examples/lid_driven_cavity"><img src="/jaxpi2/gallery/lid_driven_cavity.png" alt="Lid-driven cavity"></a>
</div>

[Browse all 16 examples →](/examples/)
