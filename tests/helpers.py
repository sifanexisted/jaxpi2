"""Shared test utilities: a tiny PINN problem for exercising the core."""

import jax
import jax.numpy as jnp
import ml_collections
from jax import grad, vmap

from jaxpi.models import ForwardIVP, create_model


def make_config(
    out_dim=1,
    loss_weights=None,
    pts_weights=None,
    causal=False,
    num_chunks=8,
    pseudo_time=False,
    schedule_free=False,
):
    config = ml_collections.ConfigDict()

    config.arch = ml_collections.ConfigDict()
    config.arch.arch_name = "Mlp"
    config.arch.num_layers = 2
    config.arch.hidden_dim = 16
    config.arch.out_dim = out_dim
    config.arch.activation = "tanh"
    config.arch.periodicity = None
    config.arch.fourier_emb = None
    config.arch.nonlinearity = 0.0

    config.optim = ml_collections.ConfigDict()
    config.optim.optimizer = "adam"
    config.optim.lr_schedule = "exponential_decay"
    config.optim.beta1 = 0.9
    config.optim.beta2 = 0.999
    config.optim.eps = 1e-8
    config.optim.learning_rate = 1e-3
    config.optim.decay_rate = 0.9
    config.optim.decay_steps = 100
    config.optim.warmup_steps = 10
    config.optim.staircase = False
    config.optim.schedule_free = schedule_free

    config.training = ml_collections.ConfigDict()
    config.training.max_steps = 30
    config.training.batch_size = 64
    config.training.num_time_windows = 2
    config.training.transfer_learning = True
    config.training.resume = False

    config.loss_weighting = ml_collections.ConfigDict()
    config.loss_weighting.strategy = "dynamic"
    config.loss_weighting.loss_weights = ml_collections.ConfigDict(
        loss_weights if loss_weights is not None else {"ics": 1.0, "res": 1.0}
    )
    config.loss_weighting.update_schedule = ml_collections.ConfigDict(
        {"start": 5, "every": 10}
    )
    config.loss_weighting.momentum = 0.9

    config.pseudo_time = ml_collections.ConfigDict()
    config.pseudo_time.enabled = pseudo_time
    config.pseudo_time.strategy = "dynamic" if pseudo_time else "constant"
    config.pseudo_time.pts_weights = ml_collections.ConfigDict(
        pts_weights if pts_weights is not None else {"res": 1.0}
    )
    config.pseudo_time.update_schedule = ml_collections.ConfigDict(
        {"start": 5, "every": 10}
    )
    config.pseudo_time.momentum = 0.9
    config.pseudo_time.shrink = ml_collections.ConfigDict()
    config.pseudo_time.shrink.enabled = False
    config.pseudo_time.shrink.start_log_drop = 3.0
    config.pseudo_time.shrink.end_log_drop = 5.0
    config.pseudo_time.shrink.min_factor = 0.1

    config.causal = ml_collections.ConfigDict()
    config.causal.enabled = causal
    config.causal.num_chunks = num_chunks
    config.causal.tol = 1.0

    config.logging = ml_collections.ConfigDict()
    config.logging.log_every_steps = 10
    config.logging.log_lr = False
    config.logging.log_losses = True
    config.logging.log_raw_losses = False
    config.logging.log_loss_weights = False
    config.logging.log_pts_weights = False
    config.logging.log_grads = False

    config.saving = ml_collections.ConfigDict()
    config.saving.num_keep_ckpts = 2
    config.saving.save_every_steps = 10

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "jaxpi-tests"
    config.wandb.name = "test-run"

    config.input_dim = 2
    config.seed = 0

    return config


class TinyIVP(ForwardIVP):
    """1D advection u_t + u_x = 0 with a supervised initial condition."""

    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)
        self.x_ics = jnp.linspace(0.0, 1.0, 32)
        self.u_ics = jnp.sin(2 * jnp.pi * self.x_ics)

    def neural_net(self, params, t, x):
        z = jnp.stack([t, x])
        return self.state.apply_fn(params, z)[0]

    def r_net(self, params, t, x):
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_x = grad(self.neural_net, argnums=2)(params, t, x)
        return u_t + u_x

    def losses(self, params, state, batch):
        u_pred = vmap(self.neural_net, (None, None, 0))(params, 0.0, self.x_ics)
        ics_loss = jnp.mean((u_pred - self.u_ics) ** 2)
        res_losses = self.compute_residual_losses(
            params,
            state,
            batch,
            pseudo_time=self.config.pseudo_time.enabled,
            causal=self.config.causal.enabled,
        )
        return {"ics": ics_loss, **res_losses}


class TwoComponentIVP(ForwardIVP):
    """Two-residual model whose r_net dict keys are NOT in alphabetical order."""

    variables = ("u", "v")

    def neural_net(self, params, t, x):
        z = jnp.stack([t, x])
        out = self.state.apply_fn(params, z)
        return out[0], out[1]

    def r_net(self, params, t, x):
        u, v = self.neural_net(params, t, x)
        # Declared non-alphabetically; v's residual is identically 3 and u's
        # identically 0, so the components are distinguishable by value.
        return {"v": 0.0 * v + 3.0, "u": 0.0 * u}

    def losses(self, params, state, batch):
        return self.compute_residual_losses(params, state, batch)


class TupleTwoComponentIVP(TwoComponentIVP):
    """Legacy-style model returning an unnamed residual tuple (must be rejected)."""

    def r_net(self, params, t, x):
        u, v = self.neural_net(params, t, x)
        return 0.0 * u, 0.0 * v + 3.0


def make_model(config, model_cls=TinyIVP):
    return create_model(config, model_cls)


def make_batch(key, n=64, t_max=1.0):
    """Random (t, x) collocation batch, time-sorted like UniformSampler."""
    batch = jax.random.uniform(key, (n, 2)) * jnp.array([t_max, 1.0])
    return batch[jnp.argsort(batch[:, 0])]
