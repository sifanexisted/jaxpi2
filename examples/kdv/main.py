# DETERMINISTIC
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from absl import app
from absl import flags

from ml_collections import config_flags

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

import models
from evaluators import KDVEvaluator
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
    # Get dataset
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0, :]

    # Define domain
    dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]])

    # Initialize residual sampler
    res_sampler = UniformSampler(dom, batch_size=config.training.batch_size)

    # Initialize model and evaluator
    model = create_model(config, models.KDV, u0=u0, t_star=t_star, x_star=x_star)
    evaluator = KDVEvaluator(config)

    train(config, model, res_sampler, evaluator=evaluator, eval_args=(u_ref,))


def main(argv):
    train_and_evaluate(FLAGS.config)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
