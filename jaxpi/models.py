from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, value_and_grad, random, tree_map, jacfwd, jacrev
from jax.tree_util import tree_map, tree_reduce, tree_leaves

from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

import optax

from jaxpi import archs
from jaxpi.utils import flatten_pytree

from soap_jax import soap


class TrainState(train_state.TrainState):
    loss_weights: Dict
    pts_weights: Dict
    momentum: float
    prev_params: Any = None

    def apply_loss_weights(self, loss_weights, **kwargs):
        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        loss_weights = tree_map(running_average, self.loss_weights, loss_weights)
        loss_weights = lax.stop_gradient(loss_weights)

        return self.replace(
            loss_weights=loss_weights,
            **kwargs,
        )

    def apply_pts_weights(self, pts_weights, **kwargs):
        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        pts_weights = tree_map(running_average, self.pts_weights, pts_weights)
        pts_weights = lax.stop_gradient(pts_weights)

        return self.replace(
            pts_weights=pts_weights,
            **kwargs,
        )


def create_arch(config):
    arch_name = config.arch_name.lower()

    if arch_name == "mlp":
        arch = archs.Mlp(**config)

    elif arch_name == "modifiedmlp":
        arch = archs.ModifiedMlp(**config)

    elif arch_name == "piratenet":
        arch = archs.PirateNet(**config)

    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def create_lr_schedule(config):
    if config.lr_schedule == "exponential_decay":
        lr = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            transition_steps=config.decay_steps,  # every decay_steps, the learning rate decays by decay_rate
            decay_rate=config.decay_rate,
            staircase=config.staircase
        )
    elif config.lr_schedule == "cosine_decay":
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,  # total number of steps for decay
            end_value=config.end_learning_rate,
        )
    return lr


def create_optimizer(config, lr):
    optimizer = config.optimizer.lower()

    if optimizer == "adam":
        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    elif optimizer == "soap":
        tx = soap(
            learning_rate=lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            weight_decay=0.0,
            precondition_frequency=2,
            max_precond_dim=10000
        )

    elif optimizer == "muon":
        tx = optax.contrib.muon(
            learning_rate=lr,
            ns_coeffs=(2, -1.5, 0.5),
            ns_steps=10,
            beta=0.99,
            adam_b1=0.99
        )

    if config.schedule_free:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.contrib.schedule_free(tx, lr, b1=config.beta1)
        )

    return tx


def create_train_state(config, tx, arch, params=None, train_state_cls=TrainState):
    # Initialize network
    x = jnp.ones(config.input_dim)
    if params is None:  # if not then, used for transfer learning
        params = arch.init(random.PRNGKey(config.seed), x)

    # if config.pseudo_time.enabled:
    pts_weights = dict(config.pseudo_time.pts_weights)
    # else:
    #     pts_weights = None

    loss_weights = dict(config.loss_weighting.loss_weights)

    state = train_state_cls.create(
        apply_fn=arch.apply,
        params=params,
        prev_params=params,
        tx=tx,
        loss_weights=loss_weights,
        pts_weights=pts_weights,
        momentum=config.loss_weighting.momentum,
    )

    return state


class PINN:
    def __init__(self, config, lr, tx, arch, state):
        self.config = config
        self.lr = lr
        self.tx = tx
        self.arch = arch
        self.state = state
        self.mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")

        self.step = self.create_step_fn()
        self.update_loss_weights = self.create_update_loss_weights_fn()
        self.update_pts_weights = self.create_update_pts_weights_fn()

        self.sol_pred_fn = vmap(self.neural_net, (None,) + (0,) * self.config.input_dim)
        self.r_pred_fn = vmap(self.r_net, (None,) + (0,) * self.config.input_dim)

    def neural_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def losses(self, params, state, batch):
        raise NotImplementedError("Subclasses should implement this!")

    @partial(jit, static_argnums=(0,))
    def compute_pts_weights(self, state, init_state, batch):
        # Unpack all columns regardless of batch dimensionality (t,x) or (t,x,y) etc.
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        # Stack predictions and residuals: shape (n_components, N)
        sols_pred = jnp.stack(self.sol_pred_fn(state.params, *coords))
        sols_prev = jnp.stack(self.sol_pred_fn(state.prev_params, *coords))

        res_pred = jnp.stack(self.r_pred_fn(state.params, *coords))
        res_prev = jnp.stack(self.r_pred_fn(state.prev_params, *coords))

        res0_perd = jnp.stack(self.r_pred_fn(init_state.params, *coords))

        if res0_perd.ndim == 1:
            res0_perd = res0_perd[None, :]
        losses0 = jnp.mean(res0_perd ** 2, axis=1)  # (n_components,)

        def cosine_decay_from_loss(
                losses,
                loss0,
                start_log_drop=3.0,  # no decay before this
                end_log_drop=5.0,  # reach min_factor here
                min_factor=0.1,
                eps=1e-8,
        ):
            log_drop = jnp.log10((loss0 + eps) / (losses + eps))
            p = jnp.clip((log_drop - start_log_drop) / (end_log_drop - start_log_drop), 0.0, 1.0)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + jnp.cos(jnp.pi * p))

        if res_pred.ndim == 1:
            sols_pred = sols_pred[None, :]  # (n_components, N)
            sols_prev = sols_prev[None, :]  # (n_components, N)
            res_pred = res_pred[None, :]
            res_prev = res_prev[None, :]

        sol_diffs = sols_pred - sols_prev
        res_diffs = res_pred - res_prev

        losses = jnp.mean(res_pred ** 2, axis=1)  # (n_components,)

        if self.config.pseudo_time.shrink.enabled:
            factors = cosine_decay_from_loss(
                losses,
                losses0,
                start_log_drop=self.config.pseudo_time.shrink.start_log_drop,
                end_log_drop=self.config.pseudo_time.shrink.end_log_drop,
                min_factor=self.config.pseudo_time.shrink.min_factor,
            )

        else:
            factors = 1.0

        weights = (
                jnp.linalg.norm(res_diffs, axis=1)
                / (jnp.linalg.norm(sol_diffs, axis=1) + 1e-8) * factors
        )
        weights = jnp.clip(weights, a_min=1e-2, a_max=100.0)
        weights = lax.stop_gradient(weights)

        keys = list(state.pts_weights.keys())
        return dict(zip(keys, weights))

    @partial(jit, static_argnums=(0,))
    def compute_loss_weights(self, state, batch):
        """
        Balance losses based on the gradient norms of each loss.
        """
        # Compute the gradient of each loss w.r.t. the parameters
        grads = jacrev(self.losses)(state.params, state, batch)

        # Compute the grad norm of each loss
        grad_norm_dict = {}
        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

        # Compute the mean of grad norms over all losses
        mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
        # Grad Norm Weighting
        w = tree_map(lambda x: (mean_grad_norm / (x + 1e-5 * mean_grad_norm)), grad_norm_dict)
        return w

    @partial(jit, static_argnums=(0,))
    def loss(self, params, state, batch):
        # Compute losses
        loss_dict = self.losses(params, state, batch)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, loss_dict, state.loss_weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss, loss_dict

    def create_step_fn(self):
        @jax.jit
        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(P(), P("batch")),
            out_specs=(P(), P(), P()),
            check_rep=False
        )
        def step(state, batch):
            prev_params = state.params
            (loss, loss_dict), grads = value_and_grad(self.loss, has_aux=True)(state.params, state, batch)
            # state = state.apply_gradients(grads=grads)
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            state = state.replace(
                step=state.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                prev_params=prev_params
            )
            return state, loss, loss_dict

        return step

    def create_update_loss_weights_fn(self):
        @jax.jit
        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(P(), P("batch")),
            out_specs=P(),
            check_rep=False
        )
        def update_loss_weights(state, batch):
            loss_weights = self.compute_loss_weights(state, batch)
            state = state.apply_loss_weights(loss_weights=loss_weights)
            return state

        return update_loss_weights

    def create_update_pts_weights_fn(self):
        @jax.jit
        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(P(), P(), P("batch")),
            out_specs=P(),
            check_rep=False
        )
        def update_pts_weights(state, prev_state, batch):
            pts_weights = self.compute_pts_weights(state, prev_state, batch)
            state = state.apply_pts_weights(pts_weights=pts_weights)
            return state

        return update_pts_weights


class ForwardIVP(PINN):
    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)
        if config.causal.enabled:
            self.tol = config.causal.tol
            self.num_chunks = config.causal.num_chunks
            self.triu = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1)

    @partial(jit, static_argnums=(0,))
    def compute_causal_weights(self, state, batch):
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        # Stack residuals: shape (n_components, N)
        res = jnp.stack(self.r_pred_fn(state.params, *coords))

        if res.ndim == 1:
            res = res[None, :]

        if self.config.pseudo_time.enabled:
            sols_pred = jnp.stack(self.sol_pred_fn(state.params, *coords))
            sols_prev = jnp.stack(self.sol_pred_fn(state.prev_params, *coords))

            pts_weights = jnp.array(list(state.pts_weights.values()))  # (n_components,)
            res = res + pts_weights[:, None] * (sols_pred - sols_prev)

        # Chunk, loss, and causal weights — all vectorised over components
        res = res.reshape(res.shape[0], self.num_chunks, -1)  # (n_components, chunks, N)
        losses = jnp.mean(res ** 2, axis=2)  # (n_components, chunks)
        gammas = lax.stop_gradient(
            jnp.exp(-self.tol * (losses @ self.triu))
        )  # (n_components, chunks)

        return gammas.min(axis=0)

    # @partial(jit, static_argnums=(0,))
    def compute_residual_losses(self, params, state, batch, pseudo_time=False, causal=False):
        keys = list(state.pts_weights.keys())  # TODO: Seperate IC/BC and PDE keys
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        res_pred = jnp.stack(self.r_pred_fn(params, *coords))  # (n_components, N)

        if res_pred.ndim == 1:
            res_pred = res_pred[None, :]

        if pseudo_time:
            sols_pred = jnp.stack(self.sol_pred_fn(params, *coords))
            sols_prev = jnp.stack(self.sol_pred_fn(state.prev_params, *coords))
            pts_weights = jnp.array(list(state.pts_weights.values()))  # (n_components,)
            res_pred = res_pred + pts_weights[:, None] * (sols_pred - sols_prev)

        if causal:
            res_pred = res_pred.reshape(res_pred.shape[0], self.num_chunks, -1)  # (n_components, chunks, n)
            chunk_loss = jnp.mean(res_pred ** 2, axis=2)  # (n_components, chunks)
            causal_weights = lax.stop_gradient(
                jnp.exp(-self.tol * (chunk_loss @ self.triu.T))
            )  # (K, chunks)
            per_key_losses = jnp.mean(chunk_loss * causal_weights, axis=1)  # (n_components,)
        else:
            per_key_losses = jnp.mean(res_pred ** 2, axis=1)  # (n_components,)

        return dict(zip(keys, per_key_losses))


class ForwardBVP(PINN):
    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)

    # @partial(jit, static_argnums=(0,))
    def compute_residual_losses(self, params, state, batch, pseudo_time=False):
        keys = list(state.pts_weights.keys())  # TODO: Seperate IC/BC and PDE keys
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        res_pred = jnp.stack(self.r_pred_fn(params, *coords))  # (n_components, N)

        if res_pred.ndim == 1:
            res_pred = res_pred[None, :]

        if pseudo_time:
            sols_pred = jnp.stack(self.sol_pred_fn(params, *coords))
            sols_prev = jnp.stack(self.sol_pred_fn(state.prev_params, *coords))
            pts_weights = jnp.array(list(state.pts_weights.values()))  # (n_components,)
            res_pred = res_pred + pts_weights[:, None] * (sols_pred - sols_prev)

        per_key_losses = jnp.mean(res_pred ** 2, axis=1)  # (n_components,)

        return dict(zip(keys, per_key_losses))
