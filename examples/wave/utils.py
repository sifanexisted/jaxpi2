import jax.numpy as jnp
from jax import vmap


def get_dataset(T=1.0, L=1.0, a=0.5, c=2, n_t=200, n_x=128):
    t_star = jnp.linspace(0, T, n_t)
    x_star = jnp.linspace(0, L, n_x)

    def u_fn(t, x):
        return jnp.sin(jnp.pi * x) * jnp.cos(c * jnp.pi * t) + \
            a * jnp.sin(2 * c * jnp.pi * x) * jnp.cos(4 * c * jnp.pi * t)

    u_exact = vmap(vmap(u_fn, (None, 0)), (0, None))(t_star, x_star)

    return u_exact, t_star, x_star
