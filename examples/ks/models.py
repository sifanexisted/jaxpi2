from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jax.experimental.jet import jet

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator


class KS(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, u0, t_star, x_star):
        super().__init__(config, lr, tx, arch, state)

        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

    def neural_net(self, params, t, x):
        t = t / self.t_star[-1]  # scale t to [0, 1]
        # x = x / self.x_star[-1] # scale x to [0, 1]
        z = jnp.stack([t, x])
        u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, t, x):
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_fn = lambda x: self.neural_net(params, t, x)
        _, (u_x, u_xx, u_xxx, u_xxxx) = jet(u_fn, (x,), [[1.0, 0.0, 0.0, 0.0]])
        return (
                u_t
                + 100.0 / 16.0 * u * u_x
                + 100.0 / 16.0 ** 2 * u_xx
                + 100.0 / 16.0 ** 4 * u_xxxx
        )

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Initial condition loss
        u_ic_pred = vmap(self.neural_net, (None, None, 0))(params, self.t0, self.x_star)
        ics_loss = jnp.mean((self.u0 - u_ic_pred) ** 2)
        res_losses = self.compute_residual_losses(params, state, batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)
        loss_dict = {"ics": ics_loss, **res_losses}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = vmap(vmap(self.neural_net, (None, None, 0)), (None, 0, None))(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class KSEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, u_ref):
        l2_error = model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def __call__(self, model, state, loss_dict, batch, u_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, u_ref)

        if self.config.causal.enabled and self.config.logging.log_causal_weights:
            causal_weights = model.compute_causal_weights(state, batch)
            self.log_dict["cas_weight"] = causal_weights.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
