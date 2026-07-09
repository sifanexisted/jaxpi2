import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

import models
from evaluators import KDVEvaluator
from utils import get_dataset


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
