# DETERMINISTIC
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from absl import app
from absl import flags

from ml_collections import config_flags

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import itertools

from jaxpi.models import create_model
from jaxpi.samplers import MeshSampler
from jaxpi.training import train

import models
from evaluators import NavierStokesEvaluator
from utils import get_dataset, inflow_profile

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def train_and_evaluate(config):
    # Get dataset
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        nu,
    ) = get_dataset()

    u_inflow, _ = inflow_profile(inflow_coords[:, 1])

    # Initialize residual sampler
    res_sampler = iter(MeshSampler(coords, batch_size=config.training.batch_size))
    if config.training.random_sampling:
        batches = res_sampler
    else:
        batches = itertools.repeat(next(res_sampler))

    # Initialize model and evaluator
    model = create_model(
        config, models.NavierStokes2D,
        u_inflow=u_inflow, inflow_coords=inflow_coords,
        outflow_coords=outflow_coords, wall_coords=wall_coords, nu=nu,
    )
    evaluator = NavierStokesEvaluator(config)

    train(config, model, batches, evaluator=evaluator, eval_args=(coords, u_ref, v_ref))


def main(argv):
    train_and_evaluate(FLAGS.config)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
