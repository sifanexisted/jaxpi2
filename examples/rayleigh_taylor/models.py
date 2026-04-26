from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator


class RayleighTaylor2D(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, t_max, alpha1, alpha2, alpha3, alpha4, ):
        super().__init__(config, lr, tx, arch, state)

        self.t_max = t_max
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

        # vmap functions
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))
        self.temp_pred_fn = vmap(vmap(self.temp_net, (None, None, 0, 0)), (None, 0, None, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.t_max
        z = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        temp = outputs[3]
        return u, v, p, temp

    def u_net(self, params, t, x, y):
        u, _, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p, _ = self.neural_net(params, t, x, y)
        return p

    def temp_net(self, params, t, x, y):
        _, _, _, temp = self.neural_net(params, t, x, y)
        return temp

    def r_net(self, params, t, x, y):
        u, v, p, temp = self.neural_net(params, t, x, y)

        ((u_t, u_x, u_y),
         (v_t, v_x, v_y),
         (_, p_x, p_y),
         (temp_t, temp_x, temp_y)) = jacrev(self.neural_net, argnums=(1, 2, 3))(params, t, x, y)

        u_hessian = hessian(self.u_net, argnums=(2, 3))(params, t, x, y)
        v_hessian = hessian(self.v_net, argnums=(2, 3))(params, t, x, y)
        temp_hessian = hessian(self.temp_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        temp_xx = temp_hessian[0][0]
        temp_yy = temp_hessian[1][1]

        ru = u_t + u * u_x + v * u_y + p_x - self.alpha1 * (u_xx + u_yy)
        rv = v_t + u * v_x + v * v_y + p_y - self.alpha1 * (v_xx + v_yy) - self.alpha2 * temp
        rc = u_x + v_y
        re = temp_t + u * temp_x + v * temp_y - self.alpha4 * (temp_xx + temp_yy)

        return ru, rv, rc, re

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Unpack batch
        ics_batch = batch["ics"]
        bcs_batch = batch["bcs"]
        res_batch = batch["res"]

        # Initial condition loss
        ics_coords, ics_labels = ics_batch
        u_batch, v_batch, temp_batch = ics_labels[:, 0], ics_labels[:, 1], ics_labels[:, 3]

        # Initial conditions loss
        u_ic_pred, v_ic_pred, _, temp_ic_pred = vmap(self.neural_net, (None, None, 0, 0))(params, 0.0,
                                                                                          ics_coords[:, 0],
                                                                                          ics_coords[:, 1])

        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        temp_ic_loss = jnp.mean((temp_ic_pred - temp_batch) ** 2)

        # Boundary condition losses
        u_bc_pred, v_bc_pred, _, temp_bc_pred = vmap(self.neural_net, (None, 0, 0, 0))(params, bcs_batch[:, 0],
                                                                                       bcs_batch[:, 1],
                                                                                       bcs_batch[:, 2])
        u_bc_loss = jnp.mean(u_bc_pred ** 2)
        v_bc_loss = jnp.mean(v_bc_pred ** 2)
        temp_bc_loss = jnp.mean(temp_bc_pred ** 2)

        res_losses = self.compute_residual_losses(params, state, res_batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "temp_ic": temp_ic_loss,
            "u_bc": u_bc_loss,
            "v_bc": v_bc_loss,
            "temp_bc": temp_bc_loss,
            **res_losses
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, coords, u_ref, v_ref, temp_ref):
        u_pred, v_pred, _, temp_pred = vmap(vmap(self.neural_net, (None, None, 0, 0)), (None, 0, None, None))(
            params, t, coords[:, 0], coords[:, 1]
        )
        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        temp_error = jnp.linalg.norm(temp_pred - temp_ref) / jnp.linalg.norm(temp_ref)

        return u_error, v_error, temp_error


class RayleighTaylor2DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, t, coords, u_ref, v_ref, temp_ref):
        u_error, v_error, temp_error = model.compute_l2_error(
            params,
            t, coords,
            u_ref,
            v_ref,
            temp_ref,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error
        self.log_dict["temp_error"] = temp_error

    def __call__(self, model, state, loss_dict, batch, t_star, mesh, u_ref, v_ref, temp_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, t_star, mesh, u_ref, v_ref, temp_ref)

        if self.config.logging.log_causal_weights:
            causal_weights = model.compute_causal_weights(state, batch['res'])
            self.log_dict["cas_weight"] = causal_weights.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
