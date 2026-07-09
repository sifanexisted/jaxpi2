"""Metric logging.

Metric names use "section/name" keys (e.g. "loss/res", "error/u",
"weights/u_ic") so that W&B groups them into separate chart sections:
losses, errors, adaptive weights, gradient norms, etc.
"""


class BaseEvaluator:
    def __init__(self, config):
        self.config = config
        self.log_dict = {}

    def log_lr(self, model, state):
        lr = model.lr(state.step)
        self.log_dict['lr'] = lr

    def log_losses(self, loss_dict):
        for key, values in loss_dict.items():
            self.log_dict["loss/" + key] = values

    def log_raw_losses(self, model, params, state, batch):
        # Unweighted residual losses (no pseudo-time shift, no causal
        # weights), sharded across devices like the training step
        res_batch = batch["res"] if isinstance(batch, dict) else batch
        loss_dict = model.compute_raw_residual_losses(params, state, res_batch)
        for key, values in loss_dict.items():
            self.log_dict["raw_loss/" + key] = values

    def log_loss_weights(self, state):
        weights = state.loss_weights
        for key, values in weights.items():
            self.log_dict["weights/" + key] = values

    def log_pts_weights(self, state):
        weights = state.pts_weights
        for key, values in weights.items():
            self.log_dict["pts_weights/" + key] = values

    def log_grads(self, model, state, batch):
        # Per-term gradient norms, sharded across devices
        grad_norms = model.compute_grad_norms(state, batch)
        for key, grad_norm in grad_norms.items():
            self.log_dict["grads/" + key] = grad_norm

    def __call__(self, model, state, loss_dict, batch, *args):
        # Initialize the log dict
        self.log_dict = {}
        params = state.params

        if self.config.logging.log_lr:
            self.log_lr(model, state)

        if self.config.logging.log_losses:
            self.log_losses(loss_dict)

        # get() keeps configs written before this flag existed working
        if self.config.logging.get("log_raw_losses", False):
            self.log_raw_losses(model, params, state, batch)

        if self.config.logging.log_loss_weights:
            self.log_loss_weights(state)

        # Always log the pseudo-time weights when the method is active
        if self.config.logging.log_pts_weights or self.config.pseudo_time.enabled:
            self.log_pts_weights(state)

        if self.config.logging.log_grads:
            self.log_grads(model, state, batch)

        return self.log_dict
