# DETERMINISTIC
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from absl import app
from absl import flags

from ml_collections import config_flags

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
from jax import vmap

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler, MeshSampler
from jaxpi.training import train_time_windows

import models
from evaluators import NavierStokes2DEvaluator
from utils import get_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def train_and_evaluate(config):
    u_ref, v_ref, w_ref, t_ref, mesh, nu = get_dataset(time_range=config.time_range)

    # Initial condition of the first time window
    ics = {"u0": u_ref[0, :], "v0": v_ref[0, :]}

    # Get the time domain for each time window
    num_time_steps = len(t_ref) // config.training.num_time_windows
    t_star = t_ref[:num_time_steps]

    # Define the time and space domain
    dt = t_star[1] - t_star[0]
    t0 = t_star[0]
    t1 = t_star[-1] + 1.1 * dt  # cover the start point of the next time window

    dom = jnp.array([[t0, t1], [0.0, 1.0], [0.0, 1.0]])

    # Initialize model and evaluator
    model = create_model(config, models.NavierStokes2D, t_max=t1, nu=nu)
    evaluator = NavierStokes2DEvaluator(config)

    def make_samplers(window_idx):
        uv_0 = jnp.stack([ics["u0"], ics["v0"]], axis=-1)
        ics_sampler = MeshSampler(mesh, uv_0, config.training.batch_size)
        res_sampler = UniformSampler(dom, config.training.batch_size)
        return {"ics": iter(ics_sampler), "res": iter(res_sampler)}

    def make_eval_args(window_idx):
        # Reference solution for the current time window
        u_star = u_ref[num_time_steps * window_idx: num_time_steps * (window_idx + 1), :]
        v_star = v_ref[num_time_steps * window_idx: num_time_steps * (window_idx + 1), :]
        w_star = w_ref[num_time_steps * window_idx: num_time_steps * (window_idx + 1), :]
        # all frames in time (the dataset stores few, coarse frames), strided in space
        return (t_star, mesh[::4], u_star[:, ::4], v_star[:, ::4], w_star[:, ::4])

    def propagate_ic(model, window_idx):
        if window_idx + 1 >= config.training.num_time_windows:
            return  # no next window to initialize
        # Predicted solution at the end of the window becomes the next IC
        ics["u0"], ics["v0"], _ = vmap(model.neural_net, in_axes=(None, None, 0, 0))(
            model.state.params, t_ref[num_time_steps], mesh[:, 0], mesh[:, 1]
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
