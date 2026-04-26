from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict

from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant, uniform

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "silu": nn.silu,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]
    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied
    trainable: Tuple[
        bool
    ]  # Specifies whether the period for each axis is trainable or not

    def setup(self):
        # Initialize period parameters as trainable or constant and store them in a flax frozen dict
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]

        self.period_params = freeze(period_params)

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    nonlinearity: Union[int, list] = 0.0

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity is not None:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)

        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = self.activation_fn(x)

        x = nn.Dense(features=self.out_dim)(x)

        return x


class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    nonlinearity: Union[int, list] = 0.0

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity is not None:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = nn.Dense(features=self.hidden_dim)(x)
        v = nn.Dense(features=self.hidden_dim)(x)

        u = self.activation_fn(u)
        v = self.activation_fn(v)

        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = nn.Dense(features=self.out_dim)(x)

        return x


class PirateBlock(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, u, v):
        identity = x

        x = nn.Dense(features=self.hidden_dim)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = nn.Dense(features=self.hidden_dim)(x)
        x = self.activation_fn(x)

        x = x * u + (1 - x) * v

        x = nn.Dense(features=self.hidden_dim)(x)
        x = self.activation_fn(x)

        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        x = alpha * x + (1 - alpha) * identity

        return x


class PirateNet(nn.Module):
    arch_name: Optional[str] = "PirateNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: Union[int, list] = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

        if isinstance(self.nonlinearity, (int, float)):
            self.nonlinearities = [self.nonlinearity] * self.num_layers
        else:
            assert len(self.nonlinearity) == self.num_layers
            self.nonlinearities = self.nonlinearity

    @nn.compact
    def __call__(self, x):
        if self.periodicity is not None:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb is not None:
            x = FourierEmbs(**self.fourier_emb)(x)

        u = nn.Dense(features=self.hidden_dim)(x)
        u = self.activation_fn(u)

        v = nn.Dense(features=self.hidden_dim)(x)
        v = self.activation_fn(v)

        for i in range(self.num_layers):
            x = PirateBlock(hidden_dim=self.hidden_dim,
                            output_dim=x.shape[-1],
                            activation=self.activation,
                            nonlinearity=self.nonlinearities[i])(x, u, v)

        x = nn.Dense(features=self.out_dim)(x)

        return x
