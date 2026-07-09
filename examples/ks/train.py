import jax.numpy as jnp
from jax import vmap

from jaxpi.models import create_model
from jaxpi.samplers import UniformSampler
from jaxpi.training import train_time_windows

import models
from evaluators import KSEvaluator
from utils import get_dataset


def train_and_evaluate(config):
    # Get the reference solution
    u_ref, t_ref, x_star = get_dataset(time_range=config.time_range)

    # Get the time domain for each time window
    num_time_steps = len(t_ref) // config.training.num_time_windows
    t_star = t_ref[:num_time_steps]  # time points for evaluation and next time windows
    u0 = u_ref[0, :]  # initial condition of the first time window

    # Define the time and space domain
    dt = t_star[1] - t_star[0]
    t0 = t_star[0]
    t1 = t_star[-1] + 1.1 * dt  # cover the start point of the next time window

    dom = jnp.array([[t0, t1], [x_star[0], x_star[-1]]])

    # Residual sampler, shared across time windows
    res_sampler = iter(UniformSampler(dom, batch_size=config.training.batch_size))

    # Initialize model and evaluator
    model = create_model(config, models.KS, u0=u0, t_star=t_star, x_star=x_star)
    evaluator = KSEvaluator(config)

    def make_samplers(window_idx):
        return res_sampler

    def make_eval_args(window_idx):
        # Reference solution for the current time window
        u_star = u_ref[num_time_steps * window_idx: num_time_steps * (window_idx + 1), :]
        return (u_star,)

    def propagate_ic(model, window_idx):
        if window_idx + 1 >= config.training.num_time_windows:
            return None  # no next window to initialize
        # Predicted solution at the end of the window becomes the next IC.
        # Rebuild the model so the new IC is picked up by the jitted step
        # functions (mutating model.u0 would not be, since jit caches trace
        # the IC as a constant).
        u0 = vmap(model.neural_net, in_axes=(None, None, 0))(
            model.state.params, t_ref[num_time_steps], x_star
        )
        return models.KS(model.config, model.lr, model.tx, model.arch, model.state,
                         u0=u0, t_star=t_star, x_star=x_star)

    train_time_windows(
        config, model, make_samplers, evaluator=evaluator,
        propagate_ic=propagate_ic, make_eval_args=make_eval_args,
    )
