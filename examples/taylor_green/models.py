from functools import partial

import jax.numpy as jnp
from jax import jit, vmap, jacrev, hessian

from jaxpi.models import ForwardIVP


class NavierStokes3D(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, t_max, nu):
        super().__init__(config, lr, tx, arch, state)

        self.t_max = t_max
        self.nu = nu
        # Residual key of each neural_net output (u, v, w, p), for pseudo-time
        self.pts_pairing = ("ru", "rv", "rw", "rc")

        # vmap functions over a spatial grid at a fixed time
        self.uvwp0_pred_fn = vmap(self.solution_net, (None, None, 0, 0, 0))
        self.vor0_pred_fn = vmap(self.vorticity_net, (None, None, 0, 0, 0))

    def neural_net(self, params, t, x, y, z):
        t = t / self.t_max
        inputs = jnp.stack([t, x, y, z])
        outputs = self.state.apply_fn(params, inputs)
        u = outputs[0]
        v = outputs[1]
        w = outputs[2]
        p = outputs[3]
        return u, v, w, p

    def solution_net(self, params, t, x, y, z):
        """The physical solution (u, v, w, p); overridden by MultiStage."""
        return self.neural_net(params, t, x, y, z)

    def u_net(self, params, t, x, y, z):
        return self.solution_net(params, t, x, y, z)[0]

    def v_net(self, params, t, x, y, z):
        return self.solution_net(params, t, x, y, z)[1]

    def w_net(self, params, t, x, y, z):
        return self.solution_net(params, t, x, y, z)[2]

    def p_net(self, params, t, x, y, z):
        return self.solution_net(params, t, x, y, z)[3]

    def vorticity_net(self, params, t, x, y, z):
        _, u_x, u_y, u_z = jacrev(self.u_net, argnums=(1, 2, 3, 4))(params, t, x, y, z)
        _, v_x, v_y, v_z = jacrev(self.v_net, argnums=(1, 2, 3, 4))(params, t, x, y, z)
        _, w_x, w_y, w_z = jacrev(self.w_net, argnums=(1, 2, 3, 4))(params, t, x, y, z)

        vor_x = w_y - v_z
        vor_y = u_z - w_x
        vor_z = v_x - u_y

        return vor_x, vor_y, vor_z

    def r_net(self, params, t, x, y, z):
        u, v, w, p = self.neural_net(params, t, x, y, z)

        ((u_t, u_x, u_y, u_z),
         (v_t, v_x, v_y, v_z),
         (w_t, w_x, w_y, w_z),
         (_, p_x, p_y, p_z)) = jacrev(self.neural_net, argnums=(1, 2, 3, 4))(params, t, x, y, z)

        u_hessian = hessian(lambda *args: self.neural_net(*args)[0], argnums=(2, 3, 4))(params, t, x, y, z)
        v_hessian = hessian(lambda *args: self.neural_net(*args)[1], argnums=(2, 3, 4))(params, t, x, y, z)
        w_hessian = hessian(lambda *args: self.neural_net(*args)[2], argnums=(2, 3, 4))(params, t, x, y, z)

        u_laplace = u_hessian[0][0] + u_hessian[1][1] + u_hessian[2][2]
        v_laplace = v_hessian[0][0] + v_hessian[1][1] + v_hessian[2][2]
        w_laplace = w_hessian[0][0] + w_hessian[1][1] + w_hessian[2][2]

        # PDE residual
        ru = u_t + u * u_x + v * u_y + w * u_z + p_x - self.nu * u_laplace
        rv = v_t + u * v_x + v * v_y + w * v_z + p_y - self.nu * v_laplace
        rw = w_t + u * w_x + v * w_y + w * w_z + p_z - self.nu * w_laplace
        rc = u_x + v_y + w_z

        return {"ru": ru, "rv": rv, "rw": rw, "rc": rc}

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Unpack batch
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        # Initial condition loss
        coords_batch, uvw_batch = ics_batch
        u_batch, v_batch, w_batch = uvw_batch[:, 0], uvw_batch[:, 1], uvw_batch[:, 2]

        u_ic_pred, v_ic_pred, w_ic_pred, _ = vmap(self.neural_net, (None, None, 0, 0, 0))(
            params, 0.0, coords_batch[:, 0], coords_batch[:, 1], coords_batch[:, 2]
        )
        u_ic_loss = jnp.mean((u_ic_pred - u_batch) ** 2)
        v_ic_loss = jnp.mean((v_ic_pred - v_batch) ** 2)
        w_ic_loss = jnp.mean((w_ic_pred - w_batch) ** 2)

        res_losses = self.compute_residual_losses(params, state, res_batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)

        loss_dict = {
            "u_ic": u_ic_loss,
            "v_ic": v_ic_loss,
            "w_ic": w_ic_loss,
            **res_losses
        }
        return loss_dict


class MultiStage(NavierStokes3D):
    """Eps-homotopy correction stage.

    The physical solution is decomposed as `sol = prev + eps * diff`, where
    `prev = sum_k eps_k * net_k` is frozen (previous stages) and `diff` is the
    trainable network. The correction is trained on the Navier-Stokes
    equations linearized around `prev`: the residual of the composed solution
    to first order in eps is `f(diff) + r(prev) / eps`, where `f` is the
    linearized operator and `r` the standard NS residual.
    """

    def __init__(self, config, lr, tx, arch, state, t_max, nu, prev_params_list, eps_list):
        # eps_list has one entry per stage including the current one
        assert len(eps_list) == len(prev_params_list) + 1
        self.prev_params_list = prev_params_list
        self.prev_eps_list = eps_list[:-1]
        self.eps = eps_list[-1]

        super().__init__(config, lr, tx, arch, state, t_max, nu)

        self.r_prev_pred_fn = vmap(self.r_prev, (0, 0, 0, 0))

    def prev_net(self, t, x, y, z):
        t = t / self.t_max
        inputs = jnp.stack([t, x, y, z])
        prevs = 0
        for params, eps in zip(self.prev_params_list, self.prev_eps_list):
            prevs += eps * self.state.apply_fn(params, inputs)

        u_prev = prevs[0]
        v_prev = prevs[1]
        w_prev = prevs[2]
        p_prev = prevs[3]
        return u_prev, v_prev, w_prev, p_prev

    def u_prev(self, t, x, y, z):
        return self.prev_net(t, x, y, z)[0]

    def v_prev(self, t, x, y, z):
        return self.prev_net(t, x, y, z)[1]

    def w_prev(self, t, x, y, z):
        return self.prev_net(t, x, y, z)[2]

    def solution_net(self, params, t, x, y, z):
        u_diff, v_diff, w_diff, p_diff = self.neural_net(params, t, x, y, z)
        u_prev, v_prev, w_prev, p_prev = self.prev_net(t, x, y, z)

        u = u_prev + self.eps * u_diff
        v = v_prev + self.eps * v_diff
        w = w_prev + self.eps * w_diff
        p = p_prev + self.eps * p_diff
        return u, v, w, p

    def r_prev(self, t, x, y, z):
        """Navier-Stokes residual of the frozen previous-stage solution."""
        u, v, w, _ = self.prev_net(t, x, y, z)

        ((u_t, u_x, u_y, u_z),
         (v_t, v_x, v_y, v_z),
         (w_t, w_x, w_y, w_z),
         (_, p_x, p_y, p_z)) = jacrev(self.prev_net, argnums=(0, 1, 2, 3))(t, x, y, z)

        u_hessian = hessian(self.u_prev, argnums=(1, 2, 3))(t, x, y, z)
        v_hessian = hessian(self.v_prev, argnums=(1, 2, 3))(t, x, y, z)
        w_hessian = hessian(self.w_prev, argnums=(1, 2, 3))(t, x, y, z)

        u_laplace = u_hessian[0][0] + u_hessian[1][1] + u_hessian[2][2]
        v_laplace = v_hessian[0][0] + v_hessian[1][1] + v_hessian[2][2]
        w_laplace = w_hessian[0][0] + w_hessian[1][1] + w_hessian[2][2]

        ru = u_t + u * u_x + v * u_y + w * u_z + p_x - self.nu * u_laplace
        rv = v_t + u * v_x + v * v_y + w * v_z + p_y - self.nu * v_laplace
        rw = w_t + u * w_x + v * w_y + w * w_z + p_z - self.nu * w_laplace
        rc = u_x + v_y + w_z

        return ru, rv, rw, rc

    def f_net(self, params, t, x, y, z):
        """NS operator linearized around the frozen previous-stage solution."""
        u_diff, v_diff, w_diff, _ = self.neural_net(params, t, x, y, z)
        u_prev, v_prev, w_prev, _ = self.prev_net(t, x, y, z)

        ((u_t_diff, u_x_diff, u_y_diff, u_z_diff),
         (v_t_diff, v_x_diff, v_y_diff, v_z_diff),
         (w_t_diff, w_x_diff, w_y_diff, w_z_diff),
         (_, p_x_diff, p_y_diff, p_z_diff)) = jacrev(self.neural_net, argnums=(1, 2, 3, 4))(params, t, x, y, z)

        u_x_prev, u_y_prev, u_z_prev = jacrev(self.u_prev, argnums=(1, 2, 3))(t, x, y, z)
        v_x_prev, v_y_prev, v_z_prev = jacrev(self.v_prev, argnums=(1, 2, 3))(t, x, y, z)
        w_x_prev, w_y_prev, w_z_prev = jacrev(self.w_prev, argnums=(1, 2, 3))(t, x, y, z)

        u_hessian = hessian(lambda *args: self.neural_net(*args)[0], argnums=(2, 3, 4))(params, t, x, y, z)
        v_hessian = hessian(lambda *args: self.neural_net(*args)[1], argnums=(2, 3, 4))(params, t, x, y, z)
        w_hessian = hessian(lambda *args: self.neural_net(*args)[2], argnums=(2, 3, 4))(params, t, x, y, z)

        u_laplace_diff = u_hessian[0][0] + u_hessian[1][1] + u_hessian[2][2]
        v_laplace_diff = v_hessian[0][0] + v_hessian[1][1] + v_hessian[2][2]
        w_laplace_diff = w_hessian[0][0] + w_hessian[1][1] + w_hessian[2][2]

        # Linearized advection terms
        u_ux = u_prev * u_x_diff + u_x_prev * u_diff
        v_uy = v_prev * u_y_diff + u_y_prev * v_diff
        w_uz = w_prev * u_z_diff + u_z_prev * w_diff

        u_vx = u_prev * v_x_diff + v_x_prev * u_diff
        v_vy = v_prev * v_y_diff + v_y_prev * v_diff
        w_vz = w_prev * v_z_diff + v_z_prev * w_diff

        u_wx = u_prev * w_x_diff + w_x_prev * u_diff
        v_wy = v_prev * w_y_diff + w_y_prev * v_diff
        w_wz = w_prev * w_z_diff + w_z_prev * w_diff

        fu = u_t_diff + u_ux + v_uy + w_uz + p_x_diff - self.nu * u_laplace_diff
        fv = v_t_diff + u_vx + v_vy + w_vz + p_y_diff - self.nu * v_laplace_diff
        fw = w_t_diff + u_wx + v_wy + w_wz + p_z_diff - self.nu * w_laplace_diff
        fc = u_x_diff + v_y_diff + w_z_diff

        return fu, fv, fw, fc

    def r_net(self, params, t, x, y, z):
        """Residual of the correction: f(diff) + r(prev) / eps.

        Minimizing its square is identical to the (f_pred + r_prev/eps)^2 loss
        of the original multi-stage formulation, and returning it from r_net
        lets MultiStage reuse compute_residual_losses (and therefore compose
        with pseudo-time stepping and causal weighting).
        """
        fu, fv, fw, fc = self.f_net(params, t, x, y, z)
        ru_prev, rv_prev, rw_prev, rc_prev = self.r_prev(t, x, y, z)

        return {
            "fu": fu + ru_prev / self.eps,
            "fv": fv + rv_prev / self.eps,
            "fw": fw + rw_prev / self.eps,
            "fc": fc + rc_prev / self.eps,
        }

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Unpack batch
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        # Initial condition loss on the correction: diff should match
        # (u0 - u0_prev) / eps at t = 0
        coords_batch, uvw_batch = ics_batch
        u_batch, v_batch, w_batch = uvw_batch[:, 0], uvw_batch[:, 1], uvw_batch[:, 2]

        u0_diff_pred, v0_diff_pred, w0_diff_pred, _ = vmap(self.neural_net, (None, None, 0, 0, 0))(
            params, 0.0, coords_batch[:, 0], coords_batch[:, 1], coords_batch[:, 2]
        )
        u0_prev, v0_prev, w0_prev, _ = vmap(self.prev_net, (None, 0, 0, 0))(
            0.0, coords_batch[:, 0], coords_batch[:, 1], coords_batch[:, 2]
        )

        u0_diff = (u_batch - u0_prev) / self.eps
        v0_diff = (v_batch - v0_prev) / self.eps
        w0_diff = (w_batch - w0_prev) / self.eps

        u0_diff_loss = jnp.mean((u0_diff_pred - u0_diff) ** 2)
        v0_diff_loss = jnp.mean((v0_diff_pred - v0_diff) ** 2)
        w0_diff_loss = jnp.mean((w0_diff_pred - w0_diff) ** 2)

        res_losses = self.compute_residual_losses(params, state, res_batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)

        loss_dict = {
            "u0_diff": u0_diff_loss,
            "v0_diff": v0_diff_loss,
            "w0_diff": w0_diff_loss,
            **res_losses
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_true_losses(self, params, batch):
        """Un-linearized IC/residual losses of the composed solution, for logging."""
        ics_batch = batch["ics"]
        res_batch = batch["res"]

        coords_batch, uvw_batch = ics_batch
        u_batch, v_batch, w_batch = uvw_batch[:, 0], uvw_batch[:, 1], uvw_batch[:, 2]

        u0_pred, v0_pred, w0_pred, _ = self.uvwp0_pred_fn(
            params, 0.0, coords_batch[:, 0], coords_batch[:, 1], coords_batch[:, 2]
        )

        r_sol_fn = vmap(self._r_solution, (None, 0, 0, 0, 0))
        ru_pred, rv_pred, rw_pred, rc_pred = r_sol_fn(
            params, res_batch[:, 0], res_batch[:, 1], res_batch[:, 2], res_batch[:, 3]
        )

        return {
            "u_ic": jnp.mean((u0_pred - u_batch) ** 2),
            "v_ic": jnp.mean((v0_pred - v_batch) ** 2),
            "w_ic": jnp.mean((w0_pred - w_batch) ** 2),
            "ru": jnp.mean(ru_pred ** 2),
            "rv": jnp.mean(rv_pred ** 2),
            "rw": jnp.mean(rw_pred ** 2),
            "rc": jnp.mean(rc_pred ** 2),
        }

    def _r_solution(self, params, t, x, y, z):
        """NS residual of the composed solution `prev + eps * diff`."""
        u, v, w, _ = self.solution_net(params, t, x, y, z)

        ((u_t, u_x, u_y, u_z),
         (v_t, v_x, v_y, v_z),
         (w_t, w_x, w_y, w_z),
         (_, p_x, p_y, p_z)) = jacrev(self.solution_net, argnums=(1, 2, 3, 4))(params, t, x, y, z)

        u_hessian = hessian(self.u_net, argnums=(2, 3, 4))(params, t, x, y, z)
        v_hessian = hessian(self.v_net, argnums=(2, 3, 4))(params, t, x, y, z)
        w_hessian = hessian(self.w_net, argnums=(2, 3, 4))(params, t, x, y, z)

        u_laplace = u_hessian[0][0] + u_hessian[1][1] + u_hessian[2][2]
        v_laplace = v_hessian[0][0] + v_hessian[1][1] + v_hessian[2][2]
        w_laplace = w_hessian[0][0] + w_hessian[1][1] + w_hessian[2][2]

        ru = u_t + u * u_x + v * u_y + w * u_z + p_x - self.nu * u_laplace
        rv = v_t + u * v_x + v * v_y + w * v_z + p_y - self.nu * v_laplace
        rw = w_t + u * w_x + v * w_y + w * w_z + p_z - self.nu * w_laplace
        rc = u_x + v_y + w_z

        return ru, rv, rw, rc
