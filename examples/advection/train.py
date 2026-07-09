import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

import models
from evaluators import Advection1DEvaluator
from utils import get_dataset


def train_and_evaluate(config):
    # Problem setup
    T = 2.0  # final time
    L = 2 * jnp.pi  # length of the domain
    c = 50  # advection speed
    n_t = 200  # number of time steps
    n_x = 128  # number of spatial points

    # Get  dataset
    u_ref, t_star, x_star = get_dataset(T, L, c, n_t, n_x)

    # Initial condition
    u0 = u_ref[0, :]

    # Define domain
    dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]])

    # Initialize residual sampler
    res_sampler = UniformSampler(dom, batch_size=config.training.batch_size)

    # Initialize model and evaluator
    model = create_model(config, models.Advection1D, u0=u0, t_star=t_star, x_star=x_star, c=c)
    evaluator = Advection1DEvaluator(config)

    train(config, model, res_sampler, evaluator=evaluator, eval_args=(u_ref,))
