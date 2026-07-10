from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP


class GinzburgLandau(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, t_max, eps, k):
        super().__init__(config, lr, tx, arch, state)

        self.t_max = t_max

        self.eps = eps
        self.k = k

        # vmap functions
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0, 0)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(self.v_net, (None, None, 0, 0)), (None, 0, None, None))

    def neural_net(self, params, t, x, y):
        t = t / self.t_max
        inputs = jnp.stack([t, x, y])
        outputs = self.state.apply_fn(params, inputs)

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

        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        v_t = grad(self.v_net, argnums=1)(params, t, x, y)

        u_hessian, v_hessian = hessian(self.neural_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        u_laplace = u_xx + u_yy
        v_laplace = v_xx + v_yy

        ru = u_t - self.eps * u_laplace - self.k * (u - u * (u ** 2 + v ** 2) + 1.5 * v * (u ** 2 + v ** 2))
        rv = v_t - self.eps * v_laplace - self.k * (v - v * (u ** 2 + v ** 2) - 1.5 * u * (u ** 2 + v ** 2))

        return {"ru": ru, "rv": rv}

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
