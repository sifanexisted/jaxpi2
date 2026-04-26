from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator


class NavierStokes2D(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, t_max, nu):
        super().__init__(config, lr, tx, arch, state)

        self.t_max = t_max
        self.nu = nu

        self.body_force_fn = lambda x, y: 2 * jnp.sin(4 * jnp.pi * y)

        # vmap functions
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))
        self.w_pred_fn = vmap(vmap(self.w_net, (None, None, 0, 0)), (None, 0, None, None))

    def neural_net(self, params, t, x, y):
        t = t / self.t_max
        z = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, z)
        u = outputs[0]
        v = outputs[1]
        p = outputs[2]
        return u, v, p

    def u_net(self, params, t, x, y):
        u, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _ = self.neural_net(params, t, x, y)
        return v

    def p_net(self, params, t, x, y):
        _, _, p = self.neural_net(params, t, x, y)
        return p

    def w_net(self, params, t, x, y):
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        w = v_x - u_y
        return w

    def r_net(self, params, t, x, y):
        u, v, p = self.neural_net(params, t, x, y)
        (u_t, u_x, u_y), (v_t, v_x, v_y), (_, p_x, p_y) = jacrev(self.neural_net, argnums=(1, 2, 3))(params, t, x, y)

        u_hessian = hessian(self.u_net, argnums=(2, 3))(params, t, x, y)
        v_hessian = hessian(self.v_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        body_force = self.body_force_fn(x, y)

        # PDE residual
        ru = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy) - body_force
        rv = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        rc = u_x + v_y

        return ru, rv, rc

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Unpack batch
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, uv_batch = ics_batch
        u_batch, v_batch = uv_batch[:, 0], uv_batch[:, 1]

        # Initial conditions loss
        u_ic_pred, v_ic_pred, _ = vmap(self.neural_net, (None, None, 0, 0))(
            params, 0.0, coords_batch[:, 0], coords_batch[:, 1]
        )
        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)

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
    def compute_l2_error(self, params, t_ref, coords, u_ref, v_ref, w_ref):
        u_pred = self.u_pred_fn(params, t_ref, coords[:, 0], coords[:, 1])
        v_pred = self.v_pred_fn(params, t_ref, coords[:, 0], coords[:, 1])
        w_pred = self.w_pred_fn(params, t_ref, coords[:, 0], coords[:, 1])

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        w_error = jnp.linalg.norm(w_pred - w_ref) / jnp.linalg.norm(w_ref)

        return u_error, v_error, w_error


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
