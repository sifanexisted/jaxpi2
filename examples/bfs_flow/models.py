from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardBVP
from jaxpi.evaluator import BaseEvaluator


class NavierStokes2D(ForwardBVP):
    def __init__(self, config, lr, tx, arch, state,
                 u_inflow,
                 inflow_coords,
                 outflow_coords,
                 wall_coords,
                 nu):
        super().__init__(config, lr, tx, arch, state)
        self.u_in = u_inflow  # inflow profile
        self.inflow_coords = inflow_coords
        self.outflow_coords = outflow_coords
        self.wall_coords = wall_coords
        self.nu = nu

        # Non-dimensionalized domain length and width
        self.L, self.W = self.wall_coords.max(axis=0) - self.wall_coords.min(axis=0)

    def neural_net(self, params, x, y):
        x = x / self.L  # rescale x into [0, 1]
        y = y / self.W  # rescale y into [0, 1]
        z = jnp.stack([x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        return u, v, p

    def u_net(self, params, x, y):
        u, _, _ = self.neural_net(params, x, y)
        return u

    def v_net(self, params, x, y):
        _, v, _ = self.neural_net(params, x, y)
        return v

    def r_net(self, params, x, y):
        u, v, p = self.neural_net(params, x, y)

        (u_x, u_y), (v_x, v_y), (p_x, p_y) = jacrev(self.neural_net, argnums=(1, 2))(params, x, y)

        u_hessian = hessian(self.u_net, argnums=(1, 2))(params, x, y)
        v_hessian = hessian(self.v_net, argnums=(1, 2))(params, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        ru = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        rv = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        rc = u_x + v_y

        return ru, rv, rc

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Inflow boundary conditions
        u_in_pred, v_in_pred, _ = vmap(self.neural_net, (None, 0, 0))(
            params, self.inflow_coords[:, 0], self.inflow_coords[:, 1])
        u_in_loss = jnp.mean((u_in_pred - self.u_in) ** 2)
        v_in_loss = jnp.mean(v_in_pred ** 2)

        # Outflow boundary conditions
        u_out_pred, v_out_pred, p_out_pred = vmap(self.neural_net, (None, 0, 0))(
            params, self.outflow_coords[:, 0], self.outflow_coords[:, 1])
        p_out_loss = jnp.mean(p_out_pred ** 2)

        # No-slip boundary conditions
        u_noslip_pred, v_noslip_pred, _ = vmap(self.neural_net, (None, 0, 0))(
            params, self.wall_coords[:, 0], self.wall_coords[:, 1])
        u_noslip_loss = jnp.mean(u_noslip_pred ** 2)
        v_noslip_loss = jnp.mean(v_noslip_pred ** 2)

        res_losses = self.compute_residual_losses(params, state, batch,
                                                  pseudo_time=self.config.pseudo_time.enabled)

        loss_dict = {
            "u_in": u_in_loss,
            "v_in": v_in_loss,
            "p_out": p_out_loss,
            "u_noslip": u_noslip_loss,
            "v_noslip": v_noslip_loss,
            **res_losses
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, coords, u_test, v_test):
        u_pred, v_pred, _ = vmap(self.neural_net, (None, 0, 0))(params, coords[:, 0], coords[:, 1])

        u_error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        v_error = jnp.linalg.norm(v_pred - v_test) / jnp.linalg.norm(v_test)

        return u_error, v_error


class NavierStokesEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, coords, u_ref, v_ref):
        u_error, v_error = model.compute_l2_error(params, coords, u_ref, v_ref)
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error

    def __call__(self, model, state, loss_dict, batch, x_star, y_star, U_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, x_star, y_star, U_ref)

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
