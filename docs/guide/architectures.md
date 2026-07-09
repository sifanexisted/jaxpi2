# Architectures

All networks are Flax modules selected by `config.arch.arch_name` and built by `create_arch`
(via `create_model`). Every architecture shares the same embedding front-end.

## Networks

### `Mlp`

A plain multilayer perceptron — the baseline.

### `ModifiedMlp`

The NTK-inspired variant of Wang et al.: two encoder streams $u, v$ gate every hidden layer,

$$
x^{(l+1)} = \sigma(W x^{(l)}) \odot u + \big(1 - \sigma(W x^{(l)})\big) \odot v,
$$

which mitigates gradient pathologies at negligible extra cost.

### `PirateNet`

Physics-informed residual adaptive networks. Each block applies two gated
transformations and a **learnable skip**:

$$
x^{(l+1)} = \alpha_l\, \mathcal{F}(x^{(l)}) + (1 - \alpha_l)\, x^{(l)},
$$

with $\alpha_l$ initialized at `config.arch.nonlinearity` (0 = identity). The network starts
shallow and deepens as training demands — the default for the hard benchmarks. Deep dive:
[PirateNets](/methods/piratenet).

::: warning PirateNet input width
The residual blocks require the embedded input to already have `hidden_dim` features — use a
Fourier embedding with `embed_dim == hidden_dim` (you get a clear error otherwise).
:::

```python
arch.arch_name = "PirateNet"
arch.num_layers = 2          # residual blocks
arch.hidden_dim = 256
arch.out_dim = 1
arch.activation = "tanh"     # relu / gelu / swish / sigmoid / tanh / sin
arch.nonlinearity = 0.0      # initial alpha
```

## Embeddings

### Fourier features

Random Fourier features lift low-dimensional inputs to a multi-scale basis and defeat spectral
bias:

$$
\gamma(x) = \big[ \cos(Bx),\, \sin(Bx) \big], \qquad B_{ij} \sim \mathcal{N}(0, s^2).
$$

```python
arch.fourier_emb = {"embed_scale": 2.0, "embed_dim": 256}
```

`embed_scale` sets the frequency content — raise it for finer solution scales (the multi-stage
Taylor–Green stages increase it per stage).

### Periodic embeddings

Exact periodic boundary conditions, for free: embed selected axes as
$(\cos(p\,x), \sin(p\,x))$ so every network output is periodic by construction — no boundary
loss needed.

```python
arch.periodicity = {
    "period": (1.0,),      # angular frequency: 2π/L for a domain of length L
    "axis": (1,),          # which input axes
    "trainable": (False,),
}
```

::: tip
`period` is an angular frequency: use `2 * jnp.pi / L` for a domain of length `L`
(so `1.0` for a `2π`-periodic domain).
:::
