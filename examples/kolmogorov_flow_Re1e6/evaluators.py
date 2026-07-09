
from jaxpi.evaluator import BaseEvaluator


class NavierStokes2DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model, state, loss_dict, batch):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_causal_weights:
            causal_weights = model.compute_causal_weights(state, batch['res'])
            self.log_dict["cas_weight"] = causal_weights.min()

        if self.config.logging.log_nonlinearities:
            for i in range(self.config.arch.num_layers):
                block = state.params['params'].get(f"PirateBlock_{i}")
                if block is not None:
                    self.log_dict[f"alpha_{i}"] = block['alpha']

        return self.log_dict
