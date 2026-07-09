# DETERMINISTIC
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from absl import app
from absl import flags

from ml_collections import config_flags

import jax

jax.config.update("jax_default_matmul_precision", "highest")

from functools import partial

from absl import logging

from jax import random
import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler, MeshSampler
from jaxpi.training import train_time_windows
from jaxpi.utils import get_eval_params

import models
from evaluators import NavierStokes2DEvaluator
from utils import get_dataset, predict_in_batches

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def predict_initial_condition(config, model, t, coords):
    """Predict (u, v, w) on the full grid at time t, for the next window's IC."""
    params = get_eval_params(model.state, config.optim.schedule_free)
    uvp_fn = partial(model.uvp0_pred_fn, params, t)
    w_fn = partial(model.w0_pred_fn, params, t)
    u, v, _ = predict_in_batches(uvp_fn, coords)
    (w,) = predict_in_batches(w_fn, coords)
    return u, v, w


def train_and_evaluate(config):
    u_ref, v_ref, w_ref, t_star, coords, nu = get_dataset(config.data_path)
    logging.info("Loaded dataset with nu = {:.1e}".format(nu))

    # Initial condition of the first time window
    ics = {
        "u0": jnp.array(u_ref[config.init_time_step]),
        "v0": jnp.array(v_ref[config.init_time_step]),
    }

    # Define the time and space domain of one time window
    dT = config.training.time_window_size
    dom = jnp.array([[0.0, 1.05 * dT], [0.0, 1.0], [0.0, 1.0]])

    # Initialize model and evaluator
    model = create_model(config, models.NavierStokes2D, t_max=dT, nu=nu)
    evaluator = NavierStokes2DEvaluator(config)

    def make_samplers(window_idx):
        # Fold the window index into the RNG keys so each window trains on a
        # different collocation sequence
        uv_0 = jnp.stack([ics["u0"], ics["v0"]], axis=-1)
        ics_sampler = MeshSampler(
            coords, uv_0, config.training.ics_batch_size,
            rng_key=random.PRNGKey(config.seed + 100 + window_idx),
        )
        res_sampler = UniformSampler(
            dom, config.training.batch_size,
            rng_key=random.PRNGKey(config.seed + 200 + window_idx),
        )
        return {"ics": iter(ics_sampler), "res": iter(res_sampler)}

    def make_eval_args(window_idx):
        # DNS snapshots that fall inside this window (in window-relative time),
        # on a subsampled grid; empty when the window contains no snapshot.
        # Physical time of window w starts at the IC snapshot's time.
        t_lo = float(t_star[config.init_time_step]) + window_idx * dT
        mask = (t_star >= t_lo - 1e-6) & (t_star <= t_lo + dT + 1e-6)
        sub = slice(None, None, 16)
        return (
            jnp.asarray(t_star[mask] - t_lo),
            coords[sub],
            jnp.asarray(u_ref[mask][:, sub]),
            jnp.asarray(v_ref[mask][:, sub]),
            jnp.asarray(w_ref[mask][:, sub]),
        )

    def propagate_ic(model, window_idx):
        # Predicted solution at t = dT becomes the next window's IC
        u0, v0, w0 = predict_initial_condition(config, model, dT, coords)
        ics["u0"], ics["v0"] = u0, v0
        enstrophy = 0.5 * jnp.mean(w0**2)
        logging.info(
            "Time window {} done, enstrophy at t = {:.3f}: {:.5f}".format(
                window_idx + 1, (window_idx + 1) * dT, enstrophy
            )
        )

    train_time_windows(
        config, model, make_samplers, evaluator=evaluator,
        propagate_ic=propagate_ic, make_eval_args=make_eval_args,
    )


def main(argv):
    train_and_evaluate(FLAGS.config)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
