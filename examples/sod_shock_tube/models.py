from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, pmap, jacrev, hessian, jacfwd

from flax import linen as nn

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator


class Euler1D(ForwardIVP):
    def __init__(self, config, lr, tx, arch, state, rho0, u0, p0, t_star, x_star, left_coords, right_coords):
        super().__init__(config, lr, tx, arch, state)

        self.rho0 = jnp.array(rho0)
        self.u0 = jnp.array(u0)
        self.p0 = jnp.array(p0)

        self.t_star = jnp.array(t_star)
        self.x_star = jnp.array(x_star)

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        self.gamma = 1.4

        self.left_coords = jnp.array(left_coords)
        self.right_coords = jnp.array(right_coords)

        self.rho_left = 1.0
        self.u_left = 0.0
        self.p_left = 1.0

        self.rho_right = 0.125
        self.u_right = 0.0
        self.p_right = 0.1

    def neural_net(self, params, t, x):
        z = jnp.stack([t, x])
        outputs = self.state.apply_fn(params, z)
        rho = outputs[0]
        u = outputs[1]
        p = outputs[2]

        rho = nn.relu(rho)
        p = nn.relu(p)
        return rho, u, p

    def U_net(self, params, t, x):
        rho, u, p = self.neural_net(params, t, x)
        E = p / (self.gamma - 1.0) + 0.5 * (rho * u ** 2)
        return rho, rho * u, E

    def F_net(self, params, t, x):
        rho, u, p = self.neural_net(params, t, x)
        E = p / (self.gamma - 1.0) + 0.5 * (rho * u ** 2)
        return rho * u, rho * u ** 2 + p, u * (E + p)

    def r_net(self, params, t, x):
        U1_t, U2_t, U3_t = jacfwd(self.U_net, argnums=1)(params, t, x)
        F1_x, F2_x, F3_x = jacfwd(self.F_net, argnums=2)(params, t, x)

        rc = U1_t + F1_x
        ru = U2_t + F2_x
        rE = U3_t + F3_x
        return rc, ru, rE

    @partial(jit, static_argnums=(0,))
    def losses(self, params, state, batch):
        # Left state boundary conditions
        rho_left_pred, u_left_pred, p_left_pred = vmap(self.neural_net, (None, 0, 0))(
            params, self.left_coords[:, 0], self.left_coords[:, 1])
        rho_left_loss = jnp.mean((rho_left_pred - self.rho_left) ** 2)
        u_left_loss = jnp.mean((u_left_pred - self.u_left) ** 2)
        p_left_loss = jnp.mean((p_left_pred - self.p_left) ** 2)

        # Right state boundary conditions
        rho_right_pred, u_right_pred, p_right_pred = vmap(self.neural_net, (None, 0, 0))(
            params, self.right_coords[:, 0], self.right_coords[:, 1])
        rho_right_loss = jnp.mean((rho_right_pred - self.rho_right) ** 2)
        u_right_loss = jnp.mean((u_right_pred - self.u_right) ** 2)
        p_right_loss = jnp.mean((p_right_pred - self.p_right) ** 2)

        # Initial condition loss
        rho_ic_pred, u_ic_pred, p_ic_pred = vmap(self.neural_net, (None, None, 0))(params, self.t0, self.x_star)
        rho_ic_loss = jnp.mean((rho_ic_pred - self.rho0) ** 2)
        u_ic_loss = jnp.mean((u_ic_pred - self.u0) ** 2)
        p_ic_loss = jnp.mean((p_ic_pred - self.p0) ** 2)

        res_losses = self.compute_residual_losses(params, state, batch,
                                                  pseudo_time=self.config.pseudo_time.enabled,
                                                  causal=self.config.causal.enabled)

        loss_dict = {
            "rho_left": rho_left_loss,
            "u_left": u_left_loss,
            "p_left": p_left_loss,
            "rho_right": rho_right_loss,
            "u_right": u_right_loss,
            "p_right": p_right_loss,
            "rho_ic": rho_ic_loss,
            "u_ic": u_ic_loss,
            "p_ic": p_ic_loss,
            **res_losses
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, rho_test, u_test, p_test):
        rho_pred, u_pred, p_pred = vmap(vmap(self.neural_net, (None, 0, None)), (None, None, 0))(params, self.t_star,
                                                                                                 self.x_star)

        rho_error = jnp.linalg.norm(rho_pred - rho_test) / jnp.linalg.norm(rho_test)
        u_error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        p_error = jnp.linalg.norm(p_pred - p_test) / jnp.linalg.norm(p_test)
        return rho_error, u_error, p_error


class Euler1DEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)

    def log_errors(self, model, params, rho_ref, u_ref, p_ref):
        rho_error, u_error, p_error = model.compute_l2_error(params, rho_ref, u_ref, p_ref)
        self.log_dict["rho_error"] = rho_error
        self.log_dict["u_error"] = u_error
        self.log_dict["p_error"] = p_error

    def __call__(self, model, state, loss_dict, batch, rho_ref, u_ref, p_ref):
        self.log_dict = super().__call__(model, state, loss_dict, batch)

        if self.config.logging.log_errors:
            self.log_errors(model, state.params, rho_ref, u_ref, p_ref)

        if self.config.causal.enabled and self.config.logging.log_causal_weights:
            causal_weight = model.compute_causal_weights(state, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [key for key in state.params['params'].keys() if
                          key.endswith(tuple([f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]))]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params['params'][key]['alpha']

        return self.log_dict
