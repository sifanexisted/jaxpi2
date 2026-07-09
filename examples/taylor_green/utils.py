import numpy as np

import jax
import jax.numpy as jnp
from jax import random, jit


def u0_fn(x, y, z):
    return jnp.sin(x) * jnp.cos(y) * jnp.cos(z)


def v0_fn(x, y, z):
    return -jnp.cos(x) * jnp.sin(y) * jnp.cos(z)


def w0_fn(x, y, z):
    return jnp.zeros_like(x)


def get_dataset(grid_res):
    """Analytical Taylor-Green initial condition on a uniform grid."""
    x_star = jnp.linspace(0, 2 * jnp.pi, grid_res)
    y_star = jnp.linspace(0, 2 * jnp.pi, grid_res)
    z_star = jnp.linspace(0, 2 * jnp.pi, grid_res)

    xx, yy, zz = jnp.meshgrid(x_star, y_star, z_star, indexing="ij")
    coords = jnp.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)

    u0 = u0_fn(coords[:, 0], coords[:, 1], coords[:, 2])
    v0 = v0_fn(coords[:, 0], coords[:, 1], coords[:, 2])
    w0 = w0_fn(coords[:, 0], coords[:, 1], coords[:, 2])

    return u0, v0, w0, coords


def get_dataset_from_pred(ic_path):
    """Initial condition from a previously predicted flow field."""
    data = np.load(ic_path, allow_pickle=True).item()

    coords = jnp.array(data["coords"])
    u0 = jnp.array(data["u"])
    v0 = jnp.array(data["v"])
    w0 = jnp.array(data["w"])

    return u0, v0, w0, coords


def predict_in_batches(pred_fn, coords, batch_size=512**2):
    """Evaluate `pred_fn(*coord_columns) -> tuple of arrays` over many points.

    The full grid (e.g. 256^3 points) is too large for a single vmap call, so
    predictions are accumulated in batches on the host.
    """
    n_points = coords.shape[0]
    pred_fn = jit(pred_fn)

    outputs = None
    for start_idx in range(0, n_points, batch_size):
        end_idx = min(start_idx + batch_size, n_points)
        batch_coords = coords[start_idx:end_idx]
        batch_outputs = pred_fn(*[batch_coords[:, i] for i in range(batch_coords.shape[1])])

        if not isinstance(batch_outputs, tuple):
            batch_outputs = (batch_outputs,)

        if outputs is None:
            outputs = [np.zeros(n_points) for _ in batch_outputs]

        for out, batch_out in zip(outputs, batch_outputs):
            out[start_idx:end_idx] = batch_out

    return tuple(jnp.array(out) for out in outputs)


def reject_sampling(key, num_samples, pdf_fn, dom, threshold, batch_size=4096):
    """Rejection-sample collocation points where `pdf_fn` is large.

    Args:
        key: JAX PRNG key.
        num_samples: Number of samples to return.
        pdf_fn: Vectorized function mapping a (batch, dim) array of points to
            (batch,) non-negative magnitudes.
        dom: (dim, 2) array of per-dimension (min, max) bounds.
        threshold: Fraction of the per-batch maximum below which points are
            always rejected.
        batch_size: Number of candidate points per iteration.

    Returns:
        (num_samples, dim) array of accepted points.
    """
    dom = jnp.asarray(dom)
    dim = dom.shape[0]

    @jit
    def propose(key):
        key, subkey1, subkey2 = random.split(key, 3)
        points = random.uniform(
            subkey1, (batch_size, dim), minval=dom[:, 0], maxval=dom[:, 1]
        )
        pdf_values = pdf_fn(points)
        pdf_max = jnp.max(pdf_values)
        u_values = random.uniform(subkey2, (batch_size,), maxval=pdf_max)
        accept = (u_values <= pdf_values) & (pdf_values >= threshold * pdf_max)
        return key, points, accept

    samples = []
    num_accepted = 0
    num_tries = 0
    while num_accepted < num_samples:
        key, points, accept = propose(key)
        accepted = points[np.asarray(accept)]
        samples.append(accepted)
        num_accepted += accepted.shape[0]
        num_tries += batch_size

    samples = jnp.concatenate(samples, axis=0)[:num_samples]
    print(f"Rejection sampling acceptance rate: {num_samples / num_tries:.4f}")
    return samples
