
from jaxpi.evaluator import BaseEvaluator


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, x_star, y_star, U_ref):
        l2_error = model.compute_l2_error(params, x_star, y_star, U_ref)
        self.log_dict["error/l2"] = l2_error

    def __call__(self, model, state, loss_dict, batch, x_star, y_star, U_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, x_star, y_star, U_ref)

        if self.config.logging.log_raw_losses:
            self.log_raw_losses(model, state.params, state, batch)  # should be res_batch

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha/{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
