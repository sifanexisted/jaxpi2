import jax.numpy as jnp


def get_dataset(time_range=[0.1, 1.0]):
    data = jnp.load("data/rayleigh_taylor.npy", allow_pickle=True).item()

    mesh = jnp.array(data["coords"])
    t_star = jnp.array(data["t"])
    start_time_step = int(time_range[0] * len(t_star))
    end_time_step = int(time_range[1] * len(t_star))
    t_star = t_star[start_time_step: end_time_step] - t_star[start_time_step]

    velocity = jnp.array(data["velocity"])[start_time_step:end_time_step]
    pressure = jnp.array(data["pressure"])[start_time_step:end_time_step]
    temperature = jnp.array(data["temperature"])[start_time_step:end_time_step]

    # parameters
    alpha1 = data["alpha1"]
    alpha2 = data["alpha2"]
    alpha3 = data["alpha3"]
    alpha4 = data["alpha4"]

    Ra = data["Ra"]
    Pr = data["Pr"]
    Ge = data["Ge"]

    return velocity, pressure, temperature, t_star, mesh, alpha1, alpha2, alpha3, alpha4, Ra, Pr, Ge
