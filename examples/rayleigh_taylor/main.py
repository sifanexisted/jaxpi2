# DETERMINISTIC
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from absl import app
from absl import flags

from ml_collections import config_flags

import jax

jax.config.update("jax_default_matmul_precision", "highest")

from functools import partial

import jax.numpy as jnp
from jax import vmap, jit, random

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler, MeshSampler, BaseSampler
from jaxpi.training import train_time_windows

import models
from evaluators import RayleighTaylor2DEvaluator
from utils import get_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


class BCSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)

        self.dom = dom

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        subkeys = random.split(key, 3)

        t = random.uniform(subkeys[0], (self.batch_size // 2,), minval=self.dom[0][0], maxval=self.dom[0][1])
        x = random.uniform(subkeys[1], (self.batch_size // 2,), minval=self.dom[1][0], maxval=self.dom[1][1])

        bc1_batch = jnp.stack([t, x, jnp.zeros_like(x)]).T
        bc2_batch = jnp.stack([t, x, 2 * jnp.ones_like(x)]).T

        bc_batch = jnp.vstack([bc1_batch, bc2_batch])

        return bc_batch


def train_and_evaluate(config):
    uv_ref, p_ref, temp_ref, t_ref, mesh, alpha1, alpha2, alpha3, alpha4, Ra, Pr, Ge = get_dataset(
        time_range=config.time_range)

    # Initial condition of the first time window
    ics = {
        "u0": uv_ref[0, :, 0],
        "v0": uv_ref[0, :, 1],
        "temp0": temp_ref[0, :],
    }

    # Get the time domain for each time window
    num_time_steps = len(t_ref) // config.training.num_time_windows
    t_star = t_ref[:num_time_steps]

    # Define the time and space domain
    dt = t_star[1] - t_star[0]
    t0 = t_star[0]
    t1 = t_star[-1] + 1.1 * dt  # cover the start point of the next time window

    dom = jnp.array([[t0, t1], [0.0, 1.0], [0.0, 2.0]])

    # Initialize model and evaluator
    model = create_model(
        config, models.RayleighTaylor2D,
        t_max=t1, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4,
    )
    evaluator = RayleighTaylor2DEvaluator(config)

    def make_samplers(window_idx):
        ics_labels = jnp.stack([ics["u0"], ics["v0"], ics["temp0"]], axis=-1)
        ics_sampler = MeshSampler(mesh, ics_labels, config.training.batch_size)
        bcs_sampler = BCSampler(dom, config.training.batch_size)
        res_sampler = UniformSampler(dom, config.training.batch_size)
        return {
            "ics": iter(ics_sampler),
            "bcs": iter(bcs_sampler),
            "res": iter(res_sampler),
        }

    def make_eval_args(window_idx):
        # Reference solution for the current time window
        u_star = uv_ref[num_time_steps * window_idx: num_time_steps * (window_idx + 1), :, 0]
        v_star = uv_ref[num_time_steps * window_idx: num_time_steps * (window_idx + 1), :, 1]
        temp_star = temp_ref[num_time_steps * window_idx: num_time_steps * (window_idx + 1), :]
        return (t_star[::4], mesh[::4], u_star[::4, ::4], v_star[::4, ::4], temp_star[::4, ::4])

    def propagate_ic(model, window_idx):
        if window_idx + 1 >= config.training.num_time_windows:
            return  # no next window to initialize
        # Predicted solution at the end of the window becomes the next IC
        ics["u0"], ics["v0"], _, ics["temp0"] = vmap(model.neural_net, in_axes=(None, None, 0, 0))(
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
