from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP


class Wave1D(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, u0, t_star, x_star, c):
        super().__init__(config, lr, tx, arch, state)

        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star
        self.c = c

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

    def neural_net(self, params, t, x):
        z = jnp.stack([t, x])
        u = self.state.apply_fn(params, z)
        return u[0]

    def u_t_net(self, params, t, x):
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        return u_t

    def r_net(self, params, t, x):
        u_tt = grad(grad(self.neural_net, argnums=1), argnums=1)(params, t, x)
        u_xx = grad(grad(self.neural_net, argnums=2), argnums=2)(params, t, x)
        return u_tt - self.c ** 2 * u_xx

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Initial condition loss
        u0_pred = vmap(self.neural_net, (None, None, 0))(params, self.t0, self.x_star)
        u0_loss = jnp.mean((self.u0 - u0_pred) ** 2)

        u_t0_pred = vmap(self.u_t_net, (None, None, 0))(params, self.t0, self.x_star)
        u_t0_loss = jnp.mean((0 - u_t0_pred) ** 2)

        # Boundary condition loss
        u_bc1_pred = vmap(self.neural_net, (None, 0, None))(params, self.t_star, self.x_star[0])
        u_bc2_pred = vmap(self.neural_net, (None, 0, None))(params, self.t_star, self.x_star[-1])
        bcs_loss = jnp.mean((u_bc1_pred) ** 2) + jnp.mean((u_bc2_pred) ** 2)

        res_losses = self.compute_residual_losses(params, state, batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)
        loss_dict = {"u0": u0_loss, "u_t0": u_t0_loss, "bcs": bcs_loss, **res_losses}
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = vmap(vmap(self.neural_net, (None, None, 0)), (None, 0, None))(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error
