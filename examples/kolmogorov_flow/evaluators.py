
from jaxpi.evaluator import BaseEvaluator


class NavierStokes2DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, t, coords, u_ref, v_ref, w_ref):
        u_error, v_error, w_error = model.compute_l2_error(
            params,
            t, coords,
            u_ref,
            v_ref,
            w_ref,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["w_error"] = w_error

    def __call__(self, model, state, loss_dict, batch, t_star, mesh, u_ref, v_ref, w_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, t_star, mesh, u_ref, v_ref, w_ref)

        if self.config.logging.log_causal_weights:
            causal_weights = model.compute_causal_weights(state, batch['res'])
            self.log_dict["cas_weight"] = causal_weights.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
