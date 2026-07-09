from functools import partial

from absl import logging

from jax import random
import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler, MeshSampler
from jaxpi.training import train_time_windows
from jaxpi.utils import get_eval_params

import models
from evaluators import NavierStokes3DEvaluator
from utils import get_dataset, get_dataset_from_pred, predict_in_batches


def get_initial_condition(config):
    if config.ic_path is not None:
        logging.info("Using predicted solution from {} as initial condition".format(config.ic_path))
        return get_dataset_from_pred(config.ic_path)
    return get_dataset(config.grid_res)


def predict_initial_condition(config, model, t, coords):
    """Predict (u, v, w) and enstrophy on the full grid at time t."""
    params = get_eval_params(model.state, config.optim.schedule_free)
    uvwp_fn = partial(model.uvwp0_pred_fn, params, t)
    vor_fn = partial(model.vor0_pred_fn, params, t)
    u, v, w, _ = predict_in_batches(uvwp_fn, coords)
    vor_x, vor_y, vor_z = predict_in_batches(vor_fn, coords)
    enstrophy = 0.5 * jnp.mean(vor_x**2 + vor_y**2 + vor_z**2)
    return u, v, w, enstrophy


def train_and_evaluate(config, workdir):
    u0, v0, w0, coords = get_initial_condition(config)
    ics = {"u0": u0, "v0": v0, "w0": w0}
    nu = 1.0 / config.Re

    # Define the time and space domain of one time window
    dT = config.training.time_window_size
    dom = jnp.array([
        [0.0, 1.05 * dT],
        [0.0, 2.0 * jnp.pi],
        [0.0, 2.0 * jnp.pi],
        [0.0, 2.0 * jnp.pi],
    ])

    # Initialize model and evaluator
    model = create_model(config, models.NavierStokes3D, t_max=dT, nu=nu)
    evaluator = NavierStokes3DEvaluator(config)

    def make_samplers(window_idx):
        # Fold the window index into the RNG keys so each window trains on a
        # different collocation sequence
        uvw_0 = jnp.stack([ics["u0"], ics["v0"], ics["w0"]], axis=-1)
        ics_sampler = MeshSampler(
            coords, uvw_0, config.training.ics_batch_size,
            rng_key=random.PRNGKey(config.seed + 100 + window_idx),
        )
        res_sampler = UniformSampler(
            dom, config.training.batch_size,
            rng_key=random.PRNGKey(config.seed + 200 + window_idx),
        )
        return {"ics": iter(ics_sampler), "res": iter(res_sampler)}

    def propagate_ic(model, window_idx):
        # Predicted solution at t = dT becomes the next window's IC
        ics["u0"], ics["v0"], ics["w0"], enstrophy = predict_initial_condition(config, model, dT, coords)
        logging.info(
            "Time window {} done, enstrophy at t = {:.3f}: {:.5f}".format(
                window_idx + 1, (window_idx + 1) * dT, enstrophy
            )
        )

    train_time_windows(config, model, make_samplers, evaluator=evaluator, propagate_ic=propagate_ic)
