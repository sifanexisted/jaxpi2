
from jaxpi.evaluator import BaseEvaluator

from models import MultiStage


class NavierStokes3DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model, state, loss_dict, batch):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_causal_weights and self.config.causal.enabled:
            causal_weights = model.compute_causal_weights(state, batch['res'])
            self.log_dict["causal/min_weight"] = causal_weights.min()

        # For multi-stage models, also log the un-linearized losses of the
        # composed solution
        if isinstance(model, MultiStage):
            true_losses = model.compute_true_losses(state.params, batch)
            for key, value in true_losses.items():
                self.log_dict["loss/" + key] = value

        if self.config.logging.log_nonlinearities:
            for i in range(self.config.arch.num_layers):
                block = state.params['params'].get(f"PirateBlock_{i}")
                if block is not None:
                    self.log_dict[f"alpha/{i}"] = block['alpha']

        return self.log_dict
