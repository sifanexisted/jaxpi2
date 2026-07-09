import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

import models
from evaluators import Wave1DEvaluator
from utils import get_dataset


def train_and_evaluate(config):
    # Problem setup
    T = 1.0  # final time
    L = 1.0  # length of the domain
    a = 0.5
    c = 2.0  # speed
    n_t = 200  # number of time steps
    n_x = 128  # number of spatial points

    # Get  dataset
    u_ref, t_star, x_star = get_dataset(T, L, a, c, n_t, n_x)

    # Initial condition
    u0 = u_ref[0, :]

    # Define domain
    dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]])

    # Initialize residual sampler
    res_sampler = UniformSampler(dom, batch_size=config.training.batch_size)

    # Initialize model and evaluator
    model = create_model(config, models.Wave1D, u0=u0, t_star=t_star, x_star=x_star, c=c)
    evaluator = Wave1DEvaluator(config)

    train(config, model, res_sampler, evaluator=evaluator, eval_args=(u_ref,))
