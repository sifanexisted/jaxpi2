import time
import os
from functools import partial
from absl import logging

import jax
from jax import vmap, jit, random
import jax.numpy as jnp

import wandb

from jaxpi.models import create_lr_schedule, create_optimizer, create_arch, create_train_state
from jaxpi.samplers import UniformSampler, MeshSampler, BaseSampler
from jaxpi.checkpointing import create_checkpoint_manager
from jaxpi.logging import Logger
from jaxpi.checkpointing import save_checkpoint, restore_checkpoint
from jaxpi.utils import create_update_scheduler

import models
from utils import get_dataset


class BCSampler(BaseSampler):
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)

        self.dom = dom

    @partial(jit, static_argnums=(0,))
    def data_generation(self, key):
        subkeys = random.split(key, 3)

        t = random.uniform(subkeys[0], (self.batch_size // 2,), minval=self.dom[0][0], maxval=self.dom[0][1])
        x = random.uniform(subkeys[1], (self.batch_size // 2,), minval=self.dom[1][0], maxval=self.dom[1][1])

        bc1_batch = jnp.stack([t, x, jnp.zeros_like(x)]).T
        bc2_batch = jnp.stack([t, x, 2 * jnp.ones_like(x)]).T

        bc_batch = jnp.vstack([bc1_batch, bc2_batch])

        return bc_batch


def train_time_window(config, model, evaluator, logger, samplers, t_star, mesh, u_star, v_star, temp_star, window_idx):
    # create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path, suffix="time_window_{}".format(window_idx))

    # Weights update scheduler
    loss_update = create_update_scheduler(**config.loss_weighting.update_schedule)
    pseudo_time_update = create_update_scheduler(**config.pseudo_time.update_schedule)

    # jit warm up
    start_time = time.time()  # Initialize before the loop
    step_offset = (window_idx - 1) * config.training.max_steps
    init_state = model.state  # Initialize prev_state for pseudo-time weights update
    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        # Sample mini-batch
        batch = {}
        for key, sampler in samplers.items():
            batch[key] = next(sampler)

        model.state, loss, loss_dict = model.step(model.state, batch)

        if config.pseudo_time.strategy == "dynamic":
            if pseudo_time_update(step):
                model.state = model.update_pts_weights(model.state, init_state, batch['res'])

        # Update weights if necessary
        if config.loss_weighting.strategy == "dynamic":
            if loss_update(step):
                model.state = model.update_loss_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                end_time = time.time()
                # Compute evaluation metrics and log to W&B
                log_dict = evaluator(model, model.state, loss_dict, batch, t_star[::4], mesh[::4],
                                     u_star[::4, ::4],
                                     v_star[::4, ::4],
                                     temp_star[::4, ::4])
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


def train_and_evaluate(config, workdir):
    uv_ref, p_ref, temp_ref, t_ref, mesh, alpha1, alpha2, alpha3, alpha4, Ra, Pr, Ge = get_dataset(
        time_range=config.time_range)

    # Initial condition of the first time window
    u0 = uv_ref[0, :, 0]
    v0 = uv_ref[0, :, 1]
    temp0 = temp_ref[0, :]

    # Get the time domain for each time window
    num_time_steps = len(t_ref) // config.training.num_time_windows
    t_star = t_ref[:num_time_steps]

    # Define the time and space domain
    dt = t_star[1] - t_star[0]
    t0 = t_star[0]
    t1 = t_star[-1] + 1.1 * dt

    x0 = 0.0
    x1 = 1.0

    y0 = 0.0
    y1 = 2.0

    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # Initialize the model
    lr = create_lr_schedule(config.optim)
    tx = create_optimizer(config.optim, lr)
    arch = create_arch(config.arch)
    state = create_train_state(config, tx, arch)

    # Initialize model
    model = models.RayleighTaylor2D(config, lr, tx, arch, state, t_max=t1, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
                                    alpha4=alpha4)

    # Initialize evaluator
    evaluator = models.RayleighTaylor2DEvaluator(config)

    # Initialize logger
    logger = Logger()

    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)
    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u_star = uv_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, 0]
        v_star = uv_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, 1]
        temp_star = temp_ref[num_time_steps * idx: num_time_steps * (idx + 1), :]

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

        # Initialize the samplers
        ics_labels = jnp.stack([u0, v0, temp0], axis=-1)
        ics_sampler = MeshSampler(mesh, ics_labels, config.training.batch_size)
        bcs_sampler = BCSampler(dom, config.training.batch_size)
        res_sampler = UniformSampler(dom, config.training.batch_size)

        samplers = {
            "ics": iter(ics_sampler),
            "bcs": iter(bcs_sampler),
            "res": iter(res_sampler),
        }

        # Training the current time window
        model = train_time_window(
            config, model, evaluator, logger, samplers, t_star, mesh, u_star, v_star, temp_star, idx + 1)

        #  Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            u0, v0, temp0 = vmap(model.neural_net, in_axes=(None, None, 0, 0))(
                model.state.params, t_ref[num_time_steps], mesh[:, 0], mesh[:, 1]
            )
