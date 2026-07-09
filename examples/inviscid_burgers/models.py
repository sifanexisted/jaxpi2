from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardIVP


class InviscidBurgers(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, u0, t_star, x_star):
        super().__init__(config, lr, tx, arch, state)

        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

    def neural_net(self, params, t, x):
        z = jnp.stack([t, x])
        u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, t, x):
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_x = grad(self.neural_net, argnums=2)(params, t, x)
        return u_t + u * u_x

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
