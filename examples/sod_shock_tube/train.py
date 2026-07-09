import jax.numpy as jnp

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train

import models
from evaluators import Euler1DEvaluator
from utils import get_dataset


def train_and_evaluate(config):
    # Get dataset
    rho_ref, u_ref, p_ref, _, _, t_star, x_star, left_coords, right_coords = get_dataset()
    rho0 = rho_ref[:, 0]
    u0 = u_ref[:, 0]
    p0 = p_ref[:, 0]

    # Define domain
    dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]])

    # Initialize residual sampler
    res_sampler = UniformSampler(dom, batch_size=config.training.batch_size)

    # Initialize model and evaluator
    model = create_model(
        config, models.Euler1D,
        rho0=rho0, u0=u0, p0=p0, t_star=t_star, x_star=x_star,
        left_coords=left_coords, right_coords=right_coords,
    )
    evaluator = Euler1DEvaluator(config)

    train(config, model, res_sampler, evaluator=evaluator, eval_args=(rho_ref, u_ref, p_ref))
