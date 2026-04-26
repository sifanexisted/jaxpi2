import time
import os
import logging

import jax
import jax.numpy as jnp
from jax import vmap

import wandb

from jaxpi.models import create_lr_schedule, create_optimizer, create_arch, create_train_state
from jaxpi.samplers import UniformSampler
from jaxpi.checkpointing import create_checkpoint_manager
from jaxpi.logging import Logger
from jaxpi.checkpointing import save_checkpoint, restore_checkpoint
from jaxpi.utils import create_update_scheduler

import models
from utils import get_dataset


def train_one_time_window(config, model, evaluator, logger, sampler, u0, u_star, window_idx):
    # Update initial condition for the current time window,
    # which is the predicted solution at the last time step of the previous time window
    model.u0 = u0

    # create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path, suffix="time_window_{}".format(window_idx))

    # Weights update scheduler
    loss_update = create_update_scheduler(**config.loss_weighting.update_schedule)
    pseudo_time_update = create_update_scheduler(**config.pseudo_time.update_schedule)

    # jit warm up
    start_time = time.time()  # Initialize before the loop
    step_offset = (window_idx - 1) * config.training.max_steps
    print("Waiting for JIT...")
    init_state = model.state  # Initialize prev_state for pseudo-time weights update
    for step in range(config.training.max_steps):
        # Sample mini-batch
        batch = next(sampler)

        model.state, loss, loss_dict = model.step(model.state, batch)

        if config.loss_weighting.strategy == "dynamic":
            if loss_update(step):
                model.state = model.update_loss_weights(model.state, batch)

        # Update weights if necessary
        if config.pseudo_time.strategy == "dynamic":
            if pseudo_time_update(step):
                model.state = model.update_pts_weights(model.state, init_state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                end_time = time.time()
                # Compute evaluation metrics and log to W&B
                log_dict = evaluator(model, model.state, loss_dict, batch, u_star)
                wandb.log(log_dict, step + step_offset)

                # Report training metrics
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = time.time()

        # Save checkpoint
        if config.saving.save_every_steps > 0:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                    step + 1
            ) == config.training.max_steps:
                save_checkpoint(ckpt_mngr, model.state)

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    if config.saving.save_every_steps > 0:
        save_checkpoint(ckpt_mngr, model.state)
        ckpt_mngr.wait_until_finished()

    return model


def train_and_evaluate(config):
    # Get the reference solution
    u_ref, t_ref, x_star = get_dataset(time_range=config.time_range)

    # Get the time domain for each time window
    num_time_steps = len(t_ref) // config.training.num_time_windows
    u_ref = u_ref[:num_time_steps, :]
    t_star = t_ref[:num_time_steps]  # time points for evaluation and next time windows
    u0 = u_ref[0, :]  # initial condition of the first time window

    # Define the time and space domain
    dt = t_star[1] - t_star[0]
    t0 = t_star[0]
    t1 = t_star[-1] + 1.1 * dt  # cover the start point of the next time window, which is t_star[num_time_steps]

    x0 = x_star[0]
    x1 = x_star[-1]
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Initialize  residual sampler
    res_sampler = iter(UniformSampler(dom, batch_size=config.training.batch_size))

    lr = create_lr_schedule(config.optim)
    tx = create_optimizer(config.optim, lr)
    arch = create_arch(config.arch)
    state = create_train_state(config, tx, arch)

    # Initialize model
    model = models.KS(config, lr, tx, arch, state, u0, t_star, x_star)

    # Initialize evaluator
    evaluator = models.KSEvaluator(config)

    # Initialize logger
    logger = Logger()

    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)
    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u_star = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :]

        if config.training.transfer_learning:
            if idx > 0:
                # restore the checkpoint from the previous time window
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path, suffix="time_window_{}".format(idx))

                state = restore_checkpoint(ckpt_mngr, model.state)
                model.state = create_train_state(
                    config,
                    tx=tx,
                    arch=arch,
                    params=state.params,

                )
                print("Restored checkpoint from previous time window {} at step {}".format(idx, state.step))

        # Training the current time window
        model = train_one_time_window(
            config, model, evaluator, logger, res_sampler, u0, u_star, idx + 1)

        #  Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            u0 = vmap(model.neural_net, in_axes=(None, None, 0))(
                model.state.params, t_ref[num_time_steps], x_star
            )
