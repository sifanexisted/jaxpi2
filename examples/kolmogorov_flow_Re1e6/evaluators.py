
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import get_eval_params


class NavierStokes2DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, model, state, loss_dict, batch, t_ref=None, coords=None,
                 u_ref=None, v_ref=None, w_ref=None):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        # Relative L2 errors against the DNS reference (subsampled snapshots
        # within the current time window), using the schedule-free averaged
        # parameters when applicable.
        if self.config.logging.log_errors and t_ref is not None and len(t_ref) > 0:
            params = get_eval_params(state, self.config.optim.schedule_free)
            u_error, v_error, w_error = model.compute_l2_error(
                params, t_ref, coords, u_ref, v_ref, w_ref
            )
            self.log_dict["u_error"] = u_error
            self.log_dict["v_error"] = v_error
            self.log_dict["w_error"] = w_error

        if self.config.logging.log_causal_weights:
            # Sharded across all devices, like the training step
            causal_weights = model.compute_causal_weights(state, batch['res'])
            self.log_dict["cas_weight"] = causal_weights.min()

        if self.config.logging.log_nonlinearities:
            for i in range(self.config.arch.num_layers):
                block = state.params['params'].get(f"PirateBlock_{i}")
                if block is not None:
                    self.log_dict[f"alpha_{i}"] = block['alpha']

        return self.log_dict
