
from jaxpi.evaluator import BaseEvaluator


class Euler1DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, rho_ref, u_ref, p_ref):
        rho_error, u_error, p_error = model.compute_l2_error(params, rho_ref, u_ref, p_ref)
        self.log_dict["error/rho"] = rho_error
        self.log_dict["error/u"] = u_error
        self.log_dict["error/p"] = p_error

    def __call__(self, model, state, loss_dict, batch, rho_ref, u_ref, p_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, rho_ref, u_ref, p_ref)

        if self.config.causal.enabled and self.config.logging.log_causal_weights:
            causal_weight = model.compute_causal_weights(state, batch)
            self.log_dict["causal/min_weight"] = causal_weight.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha/{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
