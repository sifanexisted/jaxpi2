import jax.numpy as jnp

from jax import jacrev

from jaxpi.utils import flatten_pytree


class BaseEvaluator:
    def __init__(self, config):
        self.config = config
        self.log_dict = {}

    def log_lr(self, model, state):
        lr = model.lr(state.step)
        self.log_dict['lr'] = lr

    def log_losses(self, loss_dict):
        for key, values in loss_dict.items():
            self.log_dict[key + "_loss"] = values

    def log_raw_losses(self, model, params, state, batch):
        loss_dict = model.compute_residual_losses(params, state, batch)
        for key, values in loss_dict.items():
            self.log_dict[key + "_raw_loss"] = values

    def log_loss_weights(self, state):
        weights = state.loss_weights
        for key, values in weights.items():
            self.log_dict[key + "_loss_weight"] = values

    def log_pts_weights(self, state):
        weights = state.pts_weights
        for key, values in weights.items():
            self.log_dict[key + "_pts_weight"] = values

    def log_grads(self, model, params, batch):
        grads = jacrev(model.losses)(params, model.state, batch)
        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            grad_norm = jnp.linalg.norm(flattened_grad)
            self.log_dict[key + "_grad_norm"] = grad_norm

    def __call__(self, model, state, loss_dict, batch, *args):
        # Initialize the log dict
        self.log_dict = {}
        params = state.params

        if self.config.logging.log_lr:
            self.log_lr(model, state)

        if self.config.logging.log_losses:
            self.log_losses(loss_dict)

        if self.config.logging.log_loss_weights:
            self.log_loss_weights(state)

        if self.config.logging.log_pts_weights:
            self.log_pts_weights(state)

        if self.config.logging.log_grads:
            self.log_grads(model, params, batch)

        return self.log_dict
