import itertools

import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

import models
from evaluators import InviscidBurgersEvaluator
from utils import get_dataset


def train_and_evaluate(config):
    # Get dataset
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0, :]

    # Define domain
    dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]])

    # Initialize residual sampler
    res_sampler = iter(UniformSampler(dom, batch_size=config.training.batch_size))
    if config.training.random_sampling:
        batches = res_sampler
    else:
        batches = itertools.repeat(next(res_sampler))

    # Initialize model and evaluator
    model = create_model(config, models.InviscidBurgers, u0=u0, t_star=t_star, x_star=x_star)
    evaluator = InviscidBurgersEvaluator(config)

    train(config, model, batches, evaluator=evaluator, eval_args=(u_ref,))
