import itertools
from functools import partial
from typing import Any, Dict

from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, value_and_grad, random, jacrev
from jax.tree_util import tree_map, tree_reduce, tree_leaves
from jax.sharding import PartitionSpec as P

import optax

from jaxpi import archs
from jaxpi.utils import flatten_pytree

from soap_jax import soap


def _axis_is_bound(axis_name="batch"):
    """Return True when tracing under a shard_map that binds `axis_name`.

    Lets the same loss code run both inside the sharded training step (where
    cross-device collectives are required for correctness) and outside of it
    (e.g. evaluator calls on the full, unsharded batch).
    """
    try:
        lax.axis_index(axis_name)
        return True
    except NameError:
        return False


class TrainState(train_state.TrainState):
    loss_weights: Dict
    pts_weights: Dict
    momentum: float
    pts_momentum: float
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
            lambda old_w, new_w: old_w * self.pts_momentum
            + (1 - self.pts_momentum) * new_w
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
    else:
        raise NotImplementedError(f"LR schedule {config.lr_schedule} not supported yet!")
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

    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")

    if config.schedule_free:
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.contrib.schedule_free(tx, lr, b1=config.beta1)
        )

    return tx


def create_model(config, model_cls, *model_args, params=None, **model_kwargs):
    """Build a model from its config: lr schedule, optimizer, arch, state.

    `model_args` / `model_kwargs` are forwarded to the model constructor
    (problem-specific arguments such as initial conditions or physical
    parameters). `params` warm-starts the train state (transfer learning).
    """
    lr = create_lr_schedule(config.optim)
    tx = create_optimizer(config.optim, lr)
    arch = create_arch(config.arch)
    state = create_train_state(config, tx, arch, params=params)
    return model_cls(config, lr, tx, arch, state, *model_args, **model_kwargs)


def create_train_state(config, tx, arch, params=None, train_state_cls=TrainState):
    # Initialize network
    x = jnp.ones(config.input_dim)
    if params is None:  # if not then, used for transfer learning
        params = arch.init(random.PRNGKey(config.seed), x)

    pts_weights = dict(config.pseudo_time.pts_weights)
    loss_weights = dict(config.loss_weighting.loss_weights)

    state = train_state_cls.create(
        apply_fn=arch.apply,
        params=params,
        prev_params=params,
        tx=tx,
        loss_weights=loss_weights,
        pts_weights=pts_weights,
        momentum=config.loss_weighting.momentum,
        pts_momentum=config.pseudo_time.momentum,
    )

    return state


class PINN:
    _uid_counter = itertools.count()

    def __init__(self, config, lr, tx, arch, state):
        # Methods decorated with jit(static_argnums=0) key their trace cache
        # on hash(self). The default id()-based hash can be reused after
        # garbage collection, silently resurrecting a previous model's traces
        # (with its constants, e.g. ICs or causal settings, baked in). A
        # process-unique id makes such collisions impossible.
        self._uid = next(PINN._uid_counter)

        self.config = config
        self.lr = lr
        self.tx = tx
        self.arch = arch
        self.state = state
        self.mesh = jax.make_mesh(
            (jax.device_count(),), ("batch",),
            axis_types=(jax.sharding.AxisType.Auto,),
        )

        self.step = self.create_step_fn()
        self.update_loss_weights = self.create_update_loss_weights_fn()
        self.update_pts_weights = self.create_update_pts_weights_fn()

        # Sharded evaluation helpers (safe for full multi-GPU batches)
        self.compute_raw_residual_losses = self.create_compute_raw_residual_losses_fn()
        self.compute_grad_norms = self.create_compute_grad_norms_fn()

        self.sol_pred_fn = vmap(self.sol_net, (None,) + (0,) * self.config.input_dim)
        self.r_pred_fn = vmap(self.r_net, (None,) + (0,) * self.config.input_dim)

    def __hash__(self):
        return self._uid

    def __eq__(self, other):
        return self is other

    #: Names of neural_net's outputs, in order — e.g. ("u", "v", "p").
    #: Models with multiple residual components must declare this; r_net
    #: then keys each residual by THE VARIABLE it evolves in pseudo-time
    #: (momentum-x residual -> "u", continuity -> "p", ...), and residuals,
    #: pts_weights, and solution components all pair automatically by name.
    #: Single-component models may leave it as None.
    variables = None

    def neural_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def sol_net(self, params, *args):
        """neural_net's outputs as a dict keyed by `variables`.

        With residuals keyed by variable name, the pseudo-time damping pairs
        each residual with its own solution component by key — immune to any
        ordering (JAX sorts dict keys when flattening pytrees). No override
        needed: declaring `variables` is sufficient.
        """
        outputs = self.neural_net(params, *args)
        if self.variables is None:
            return outputs  # single-component models
        return dict(zip(self.variables, outputs))

    def losses(self, params, state, batch):
        raise NotImplementedError("Subclasses should implement this!")

    def _stack_residuals(self, res, state):
        """Name and stack r_net outputs into keys and a (n_components, N) array.

        Multi-component r_net implementations must return a dict keyed by
        the VARIABLE each residual evolves in pseudo-time (e.g. {"u": ru,
        "v": rv, "p": rc}). A bare array (or 1-tuple) is also supported for
        single-component problems and is labeled with the single pts_weights
        key. Unnamed multi-component returns are rejected: matching them to
        pts_weights by dict iteration order (alphabetical for ConfigDict)
        silently mislabels losses and misapplies pseudo-time weights.

        Everything downstream (pts weights, the damping term's solution
        components from sol_net) is matched to these keys BY NAME, so row
        order is irrelevant.
        """
        if isinstance(res, dict):
            keys = list(res.keys())
            assert set(keys) == set(state.pts_weights.keys()), (
                f"r_net returned residual keys {keys}, but pts_weights has "
                f"keys {list(state.pts_weights.keys())}; they must match"
            )
            res = jnp.stack([res[key] for key in keys])
        else:
            keys = list(state.pts_weights.keys())
            res = jnp.stack(res)
            if res.ndim == 1:
                res = res[None, :]
            assert res.shape[0] == 1 and len(keys) == 1, (
                f"r_net returned {res.shape[0]} unnamed residual components "
                f"for pts_weights keys {keys}. Return a dict from r_net "
                'keyed by variable (e.g. {"u": ru, "v": rv, "p": rc}) so '
                "that components are matched to their weights by name"
            )
        return keys, res

    def _pts_weight_vector(self, keys, state):
        """Pseudo-time weights as a vector in residual-component order."""
        return jnp.array([state.pts_weights[key] for key in keys])

    @staticmethod
    def _residual_loss_names(keys):
        """Loss-dict names for residual components: variable-keyed residuals
        get a `_res` suffix ("u" -> "u_res") so losses are self-identifying
        next to e.g. "u_ic"; the plain single-component "res" stays as-is."""
        return [key if key == "res" else f"{key}_res" for key in keys]

    def _sols_matrix(self, sols, keys):
        """sol_net output as a (n_components, N) matrix whose rows follow
        the residual-component order `keys` — matched by variable name.
        Single-component models may return a bare array (or 1-tuple)."""
        if isinstance(sols, dict):
            missing = set(keys) - set(sols.keys())
            assert not missing, (
                f"residuals are keyed by variables {keys}, but sol_net "
                f"returned no component for {sorted(missing)} (variables "
                f"declared: {self.variables})"
            )
            return jnp.stack([sols[key] for key in keys])

        if not isinstance(sols, (tuple, list)):
            sols = (sols,)
        assert len(sols) == 1 and len(keys) == 1, (
            f"model has {len(keys)} residual components {keys} but does not "
            "declare `variables`. Set e.g. variables = (\"u\", \"v\", \"p\") "
            "naming neural_net's outputs, and key r_net's residuals by the "
            "variable each one evolves in pseudo-time"
        )
        return jnp.stack(sols)

    def compute_pts_weights(self, state, init_state, batch):
        # Unpack all columns regardless of batch dimensionality (t,x) or (t,x,y) etc.
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        keys, res_pred = self._stack_residuals(self.r_pred_fn(state.params, *coords), state)
        _, res_prev = self._stack_residuals(self.r_pred_fn(state.prev_params, *coords), state)
        _, res0_pred = self._stack_residuals(self.r_pred_fn(init_state.params, *coords), state)

        # Solution components aligned with the residual rows BY NAME
        sols_pred = self._sols_matrix(self.sol_pred_fn(state.params, *coords), keys)
        sols_prev = self._sols_matrix(self.sol_pred_fn(state.prev_params, *coords), keys)

        # Reductions must be global when the batch is sharded across devices,
        # so that all devices agree on the resulting weights.
        if _axis_is_bound("batch"):
            global_mean = lambda x: lax.pmean(x, "batch")
            global_norm = lambda x: jnp.sqrt(
                lax.psum(jnp.sum(x**2, axis=1), "batch")
            )
        else:
            global_mean = lambda x: x
            global_norm = lambda x: jnp.linalg.norm(x, axis=1)

        losses0 = global_mean(jnp.mean(res0_pred ** 2, axis=1))  # (n_components,)

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

        sol_diffs = sols_pred - sols_prev
        res_diffs = res_pred - res_prev

        losses = global_mean(jnp.mean(res_pred ** 2, axis=1))  # (n_components,)

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
                global_norm(res_diffs)
                / (global_norm(sol_diffs) + 1e-8) * factors
        )
        weights = jnp.clip(weights, 1e-2, 100.0)
        weights = lax.stop_gradient(weights)

        return dict(zip(keys, weights))

    def _grad_norms(self, state, batch):
        """Per-loss-term gradient norms on the (possibly sharded) batch."""
        grads = jacrev(self.losses)(state.params, state, batch)

        # Average gradients over the sharded batch so all devices agree
        if _axis_is_bound("batch"):
            grads = lax.pmean(grads, "batch")

        grad_norm_dict = {}
        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)
        return grad_norm_dict

    def compute_loss_weights(self, state, batch):
        """
        Balance losses based on the gradient norms of each loss.
        """
        grad_norm_dict = self._grad_norms(state, batch)

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

    def _shard_batch(self, batch):
        """Constrain the batch to be sharded along the leading (batch) axis."""
        sharding = jax.NamedSharding(self.mesh, P("batch"))
        return tree_map(
            lambda x: lax.with_sharding_constraint(x, sharding), batch
        )

    def _replicate(self, tree):
        """Constrain a pytree (e.g. the train state) to be fully replicated."""
        sharding = jax.NamedSharding(self.mesh, P())
        return tree_map(
            lambda x: lax.with_sharding_constraint(x, sharding), tree
        )

    def create_step_fn(self):
        @partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(P(), P("batch")),
            out_specs=(P(), P(), P()),
            check_vma=False
        )
        def sharded_step(state, batch):
            prev_params = state.params
            (loss, loss_dict), grads = value_and_grad(self.loss, has_aux=True)(state.params, state, batch)
            # Average the loss and gradients over the sharded batch; without
            # this each device would apply a different update and the
            # replicated parameters would silently diverge.
            loss = lax.pmean(loss, "batch")
            loss_dict = lax.pmean(loss_dict, "batch")
            grads = lax.pmean(grads, "batch")
            updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            state = state.replace(
                step=state.step + 1,
                params=new_params,
                opt_state=new_opt_state,
                prev_params=prev_params
            )
            return state, loss, loss_dict

        @jax.jit
        def step(state, batch):
            return sharded_step(self._replicate(state), self._shard_batch(batch))

        return step

    def create_update_loss_weights_fn(self):
        @partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(P(), P("batch")),
            out_specs=P(),
            check_vma=False
        )
        def sharded_update(state, batch):
            loss_weights = self.compute_loss_weights(state, batch)
            state = state.apply_loss_weights(loss_weights=loss_weights)
            return state

        @jax.jit
        def update_loss_weights(state, batch):
            return sharded_update(self._replicate(state), self._shard_batch(batch))

        return update_loss_weights

    def create_compute_raw_residual_losses_fn(self):
        """Sharded unweighted residual losses (no pseudo-time, no causal) —
        used by evaluators, so that full multi-GPU batches fit in memory."""
        @partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(P(), P(), P("batch")),
            out_specs=P(),
            check_vma=False,
        )
        def sharded_losses(params, state, batch):
            losses = self.compute_residual_losses(params, state, batch)
            return lax.pmean(losses, "batch")

        @jax.jit
        def compute_raw_residual_losses(params, state, batch):
            return sharded_losses(
                self._replicate(params), self._replicate(state), self._shard_batch(batch)
            )

        return compute_raw_residual_losses

    def create_compute_grad_norms_fn(self):
        """Sharded per-term gradient norms — used by evaluators."""
        @partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(P(), P("batch")),
            out_specs=P(),
            check_vma=False,
        )
        def sharded_norms(state, batch):
            return self._grad_norms(state, batch)

        @jax.jit
        def compute_grad_norms(state, batch):
            return sharded_norms(self._replicate(state), self._shard_batch(batch))

        return compute_grad_norms

    def create_update_pts_weights_fn(self):
        @partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(P(), P(), P("batch")),
            out_specs=P(),
            check_vma=False
        )
        def sharded_update(state, prev_state, batch):
            pts_weights = self.compute_pts_weights(state, prev_state, batch)
            state = state.apply_pts_weights(pts_weights=pts_weights)
            return state

        @jax.jit
        def update_pts_weights(state, prev_state, batch):
            return sharded_update(
                self._replicate(state), self._replicate(prev_state), self._shard_batch(batch)
            )

        return update_pts_weights


class ForwardIVP(PINN):
    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)
        self.tol = config.causal.tol
        self.num_chunks = config.causal.num_chunks
        self.triu = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1)

        # Sharded like the training step, so evaluators can compute the exact
        # causal gates on the full (multi-GPU) batch without OOM-ing a device.
        self.compute_causal_weights = self.create_compute_causal_weights_fn()

    def _global_chunk_losses(self, res_pred):
        """Global per-chunk residual losses in time order: (n_components, num_chunks).

        `res_pred` is this device's (n_components, N_local) slice of the
        globally time-sorted residuals (or the full batch outside shard_map).
        """
        num_devices = self.mesh.shape["batch"] if _axis_is_bound("batch") else 1

        assert self.num_chunks % num_devices == 0, (
            f"causal.num_chunks={self.num_chunks} must be divisible by the "
            f"number of devices {num_devices}"
        )
        local_chunks = self.num_chunks // num_devices

        assert res_pred.shape[1] % local_chunks == 0, (
            f"Residual batch of {res_pred.shape[1]} points (per device) is not "
            f"divisible by its {local_chunks} local causal chunks"
        )

        # (n_components, local_chunks)
        res_pred = res_pred.reshape(res_pred.shape[0], local_chunks, -1)
        chunk_loss = jnp.mean(res_pred**2, axis=2)

        if num_devices > 1:
            # concatenated in device order == global time order
            chunk_loss = lax.all_gather(chunk_loss, "batch", axis=1, tiled=True)
        return chunk_loss

    def _causal_residuals(self, params, state, batch, pseudo_time):
        """Residuals (with optional pseudo-time shift) for causal chunking."""
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))
        keys, res = self._stack_residuals(self.r_pred_fn(params, *coords), state)

        if pseudo_time:
            sols_pred = self._sols_matrix(self.sol_pred_fn(params, *coords), keys)
            sols_prev = self._sols_matrix(self.sol_pred_fn(state.prev_params, *coords), keys)
            pts_weights = self._pts_weight_vector(keys, state)  # (n_components,)
            res = res + pts_weights[:, None] * (sols_pred - sols_prev)
        return res

    def create_compute_causal_weights_fn(self):
        @partial(
            jax.shard_map,
            mesh=self.mesh,
            in_specs=(P(), P("batch")),
            out_specs=P(),
            check_vma=False,
        )
        def sharded_weights(state, batch):
            res = self._causal_residuals(
                state.params, state, batch, self.config.pseudo_time.enabled
            )
            chunk_loss = self._global_chunk_losses(res)  # (n_components, num_chunks)
            gammas = lax.stop_gradient(
                jnp.exp(-self.tol * (chunk_loss @ self.triu))
            )
            return gammas.min(axis=0)

        @jax.jit
        def compute_causal_weights(state, batch):
            return sharded_weights(self._replicate(state), self._shard_batch(batch))

        return compute_causal_weights

    def _causal_losses(self, res_pred):
        """Causally weighted per-component losses for time-sorted residuals.

        `res_pred` has shape (n_components, N) with points sorted by time. The
        *global* batch is split into `causal.num_chunks` chunks and each chunk
        is gated by the cumulative loss of all *earlier* chunks. Under
        shard_map the batch is time-sorted globally and split contiguously
        across devices, so per-device chunk losses concatenated in device order
        recover the global time ordering. Each device returns its local share
        of the global loss, so that pmean over devices yields the global
        causal loss (and its exact gradient) for any device count.
        """
        num_devices = self.mesh.shape["batch"] if _axis_is_bound("batch") else 1
        local_chunks = self.num_chunks // num_devices

        chunk_loss_global = self._global_chunk_losses(res_pred)

        if num_devices > 1:
            gammas = lax.stop_gradient(
                jnp.exp(-self.tol * (chunk_loss_global @ self.triu))
            )  # (n_components, num_chunks)
            # This device's slice of the global causal weights
            start = lax.axis_index("batch") * local_chunks
            gammas = lax.dynamic_slice_in_dim(gammas, start, local_chunks, axis=1)
            # this device's local chunk losses (own slice of the gathered array)
            chunk_loss = lax.dynamic_slice_in_dim(
                chunk_loss_global, start, local_chunks, axis=1
            )
        else:
            chunk_loss = chunk_loss_global
            gammas = lax.stop_gradient(
                jnp.exp(-self.tol * (chunk_loss_global @ self.triu))
            )  # (n_components, num_chunks)

        return jnp.mean(chunk_loss * gammas, axis=1)

    def compute_residual_losses(self, params, state, batch, pseudo_time=False, causal=False):
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        keys, res_pred = self._stack_residuals(self.r_pred_fn(params, *coords), state)

        if pseudo_time:
            sols_pred = self._sols_matrix(self.sol_pred_fn(params, *coords), keys)
            sols_prev = self._sols_matrix(self.sol_pred_fn(state.prev_params, *coords), keys)
            pts_weights = self._pts_weight_vector(keys, state)  # (n_components,)
            res_pred = res_pred + pts_weights[:, None] * (sols_pred - sols_prev)

        if causal:
            per_key_losses = self._causal_losses(res_pred)  # (n_components,)
        else:
            per_key_losses = jnp.mean(res_pred ** 2, axis=1)  # (n_components,)

        return dict(zip(self._residual_loss_names(keys), per_key_losses))


class ForwardBVP(PINN):
    def __init__(self, config, lr, tx, arch, state):
        super().__init__(config, lr, tx, arch, state)

    def compute_residual_losses(self, params, state, batch, pseudo_time=False):
        coords = tuple(batch[:, i] for i in range(batch.shape[1]))

        keys, res_pred = self._stack_residuals(self.r_pred_fn(params, *coords), state)

        if pseudo_time:
            sols_pred = self._sols_matrix(self.sol_pred_fn(params, *coords), keys)
            sols_prev = self._sols_matrix(self.sol_pred_fn(state.prev_params, *coords), keys)
            pts_weights = self._pts_weight_vector(keys, state)  # (n_components,)
            res_pred = res_pred + pts_weights[:, None] * (sols_pred - sols_prev)

        per_key_losses = jnp.mean(res_pred ** 2, axis=1)  # (n_components,)

        return dict(zip(self._residual_loss_names(keys), per_key_losses))
