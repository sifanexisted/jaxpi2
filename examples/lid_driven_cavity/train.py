import itertools

import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

import models
from evaluators import NavierStokesEvaluator
from utils import get_dataset


def train_and_evaluate(config):
    # Get dataset
    u_ref, v_ref, x_star, y_star, nu = get_dataset(config.Re)
    U_ref = jnp.sqrt(u_ref ** 2 + v_ref ** 2)

    # Define domain
    dom = jnp.array([[x_star[0], x_star[-1]], [y_star[0], y_star[-1]]])

    # Initialize residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size))
    if config.training.random_sampling:
        batches = res_sampler
    else:
        batches = itertools.repeat(next(res_sampler))

    # Initialize model and evaluator
    model = create_model(config, models.NavierStokes2D, nu=1 / config.Re)
    evaluator = NavierStokesEvaluator(config)

    train(config, model, batches, evaluator=evaluator, eval_args=(x_star, y_star, U_ref))
