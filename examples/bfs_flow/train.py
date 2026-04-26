import time
import os

import jax
import jax.numpy as jnp

import wandb

from jaxpi.models import create_lr_schedule, create_optimizer, create_arch, create_train_state
from jaxpi.samplers import MeshSampler
from jaxpi.checkpointing import create_checkpoint_manager
from jaxpi.logging import Logger
from jaxpi.checkpointing import save_checkpoint
from jaxpi.utils import create_update_scheduler

import models
from utils import get_dataset, inflow_profile


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

    # Initialize  residual sampler
    res_sampler = iter(MeshSampler(coords, batch_size=config.training.batch_size))

    lr = create_lr_schedule(config.optim)
    tx = create_optimizer(config.optim, lr)
    arch = create_arch(config.arch)
    state = create_train_state(config, tx, arch)

    # Initialize model
    model = models.NavierStokes2D(config, lr, tx, arch, state, u_inflow, inflow_coords, outflow_coords, wall_coords, nu)

    # Initialize evaluator
    evaluator = models.NavierStokesEvaluator(config)

    # Weights update scheduler
    loss_update = create_update_scheduler(**config.loss_weighting.update_schedule)
    pseudo_time_update = create_update_scheduler(**config.pseudo_time.update_schedule)

    # Initialize logger
    logger = Logger()

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # jit warm up
    start_time = time.time()  # Initialize before the loop
    print("Waiting for JIT...")

    if not config.training.random_sampling:
        fixed_batch = next(res_sampler)

    init_state = model.state  # Initialize prev_state for pseudo-time weights update
    for step in range(config.training.max_steps):
        batch = next(res_sampler) if config.training.random_sampling else fixed_batch
        model.state, loss, loss_dict = model.step(model.state, batch)

        if config.pseudo_time.strategy == "dynamic":
            if pseudo_time_update(step):
                model.state = model.update_pts_weights(model.state, init_state, batch)

        # Update weights if necessary
        if config.loss_weighting.strategy == "dynamic":
            if loss_update(step):
                model.state = model.update_loss_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                end_time = time.time()
                # Compute evaluation metrics and log to W&B
                log_dict = evaluator(model, model.state, loss_dict, batch, coords, u_ref, v_ref)
                wandb.log(log_dict, step)

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
