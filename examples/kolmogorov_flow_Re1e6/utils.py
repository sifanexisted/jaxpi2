import numpy as np

import jax.numpy as jnp
from jax import jit


def get_dataset(data_path):
    data = np.load(data_path, allow_pickle=True).item()

    w_ref = np.array(data["vorticity"])
    velocity = np.array(data["velocity"])

    u_ref = velocity[..., 0]
    v_ref = velocity[..., 1]

    t_star = jnp.array(data["t"]).flatten()
    t_star = t_star - t_star[0]

    coords = jnp.array(data["coords"])
    nu = float(data["nu"])

    return u_ref, v_ref, w_ref, t_star, coords, nu


def predict_in_batches(pred_fn, coords, batch_size=512**2):
    """Evaluate `pred_fn(x, y) -> tuple of arrays` over a large set of points.

    The full grid (2048^2 points) is too large for a single vmap call, so
    predictions are accumulated in batches on the host.
    """
    n_points = coords.shape[0]
    pred_fn = jit(pred_fn)

    outputs = None
    for start_idx in range(0, n_points, batch_size):
        end_idx = min(start_idx + batch_size, n_points)
        batch_outputs = pred_fn(coords[start_idx:end_idx, 0], coords[start_idx:end_idx, 1])

        if not isinstance(batch_outputs, tuple):
            batch_outputs = (batch_outputs,)

        if outputs is None:
            outputs = [np.zeros(n_points) for _ in batch_outputs]

        for out, batch_out in zip(outputs, batch_outputs):
            out[start_idx:end_idx] = batch_out

    return tuple(jnp.array(out) for out in outputs)
