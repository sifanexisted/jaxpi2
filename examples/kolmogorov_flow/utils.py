import jax.numpy as jnp


def get_dataset(time_range=[0.0, 1.0]):
    data = jnp.load("data/kolmogorov_flow_Re10000.npy", allow_pickle=True).item()
    w_ref = jnp.array(data["vorticity"])
    velocity = jnp.array(data["velocity"])

    u_ref = velocity[..., 0]
    v_ref = velocity[..., 1]

    t_star = jnp.array(data["t"]).flatten()

    start_time_step = int(time_range[0] * len(t_star))
    end_time_step = int(time_range[1] * len(t_star))
    num_time_steps = end_time_step - start_time_step
    t_star = t_star[:num_time_steps] - t_star[start_time_step]

    # Truncate data
    u_ref = u_ref[start_time_step:end_time_step]
    v_ref = v_ref[start_time_step:end_time_step]
    w_ref = w_ref[start_time_step:end_time_step]

    coords = jnp.array(data["coords"])
    nu = data["nu"]

    return u_ref, v_ref, w_ref, t_star, coords, nu
