from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian

from jaxpi.models import ForwardBVP

from utils import sample_points_on_square_boundary


class NavierStokes2D(ForwardBVP):
    def __init__(self, config, lr, tx, arch, state, nu):
        super().__init__(config, lr, tx, arch, state)
        self.nu = nu

        # Sample boundary points uniformly
        num_pts = 128

        self.x_bc1 = sample_points_on_square_boundary(num_pts,
                                                      eps=0.005)  # avoid singularity a right corner for u velocity
        self.x_bc2 = sample_points_on_square_boundary(num_pts, eps=0.005)

        # Boundary conditions
        self.v_bc = jnp.zeros((num_pts * 4,))
        self.u_bc = self.v_bc.at[:num_pts].set(1.0)

        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.v_pred_fn = vmap(self.v_net, (None, 0, 0))
        self.p_pred_fn = vmap(self.p_net, (None, 0, 0))

    # Names of neural_net's outputs; residuals are keyed by the variable
    # they evolve in pseudo-time
    variables = ("u", "v", "p")

    def neural_net(self, params, x, y):
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

    def p_net(self, params, x, y):
        _, _, p = self.neural_net(params, x, y)
        return p

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

        return {"u": ru, "v": rv, "p": rc}

    def diffusion_net(self, params, x, y):
        u_hessian = hessian(self.u_net, argnums=(1, 2))(params, x, y)
        v_hessian = hessian(self.v_net, argnums=(1, 2))(params, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        return u_xx, u_yy, v_xx, v_yy

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # boundary condition losses
        # Compute forward pass of u and v
        u_pred = self.u_pred_fn(params, self.x_bc1[:, 0], self.x_bc1[:, 1])
        v_pred = self.v_pred_fn(params, self.x_bc2[:, 0], self.x_bc2[:, 1])

        # Compute losses
        u_bc_loss = jnp.mean((u_pred - self.u_bc) ** 2)
        v_bc_loss = jnp.mean(v_pred ** 2)

        res_losses = self.compute_residual_losses(params, state, batch, pseudo_time=self.config.pseudo_time.enabled)

        loss_dict = {
            "u_bc": u_bc_loss,
            "v_bc": v_bc_loss,
            **res_losses
        }

        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, x_star, y_star, U_test):
        u_pred = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        v_pred = vmap(vmap(self.v_net, (None, None, 0)), (None, 0, None))(
            params, x_star, y_star
        )
        U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)
        l2_error = jnp.linalg.norm(U_pred - U_test) / jnp.linalg.norm(U_test)
        return l2_error
