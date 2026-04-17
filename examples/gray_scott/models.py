from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator


class GrayScott(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, t_max, b1, b2, c1, c2, eps1, eps2):
        super().__init__(config, lr, tx, arch, state)

        self.t_max = t_max

        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.c2 = c2
        self.eps1 = eps1
        self.eps2 = eps2

        # vmap functions
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))

    def neural_net(self, params, t, x, y):
        t = t / self.t_max
        z = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, z)

        u = outputs[0]
        v = outputs[1]
        return u, v

    def u_net(self, params, t, x, y):
        u, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v = self.neural_net(params, t, x, y)
        return v

    def r_net(self, params, t, x, y):
        u, v = self.neural_net(params, t, x, y)
        u_t, v_t = jacrev(self.neural_net, argnums=1)(params, t, x, y)
        u_hessian, v_hessian = hessian(self.neural_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        ru = u_t - self.b1 * (1 - u) + self.c1 * u * v ** 2 - self.eps1 * (u_xx + u_yy)
        rv = v_t + self.b2 * v - self.c2 * u * v ** 2 - self.eps2 * (v_xx + v_yy)

        return ru, rv

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Unpack batch
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, uv_batch = ics_batch
        u_batch, v_batch = uv_batch[:, 0], uv_batch[:, 1]

        # Initial conditions loss
        u_ic_pred, v_ic_pred = vmap(self.neural_net, (None, None, 0, 0))(
            params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )
        u_ic_loss = jnp.mean(jnp.abs(u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean(jnp.abs(v_ic_pred - v_batch) ** 2)

        res_losses = self.compute_residual_losses(params, state, res_batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            **res_losses
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t_ref, coords, u_ref, v_ref):
        u_pred = self.u_pred_fn(params, t_ref, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, t_ref, coords[:, 0], coords[:, 1])

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)

        return u_error, v_error


class GrayScottEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, t, coords, u_ref, v_ref):
        u_error, v_error = model.compute_l2_error(
            params,
            t, coords,
            u_ref,
            v_ref,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error

    def __call__(self, model, state, loss_dict, batch, t_star, mesh, u_ref, v_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, t_star, mesh, u_ref, v_ref)

        if self.config.logging.log_causal_weights:
            causal_weights = model.compute_causal_weights(state, batch['res'])
            self.log_dict["cas_weight"] = causal_weights.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
