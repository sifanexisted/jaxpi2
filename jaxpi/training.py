"""Shared training loops.

The inner step loop is identical for boundary-value and time-dependent
problems (that distinction lives in ForwardBVP / ForwardIVP), so there is a
single `train_loop`. Time-dependent problems trained forward in time over
consecutive windows use the `train_time_windows` orchestrator on top of it.
"""

import time

from absl import logging

import jax
import wandb

from jaxpi.checkpointing import (
    create_checkpoint_manager,
    get_ckpt_path,
    latest_time_window,
    restore_checkpoint,
    save_checkpoint,
)
from jaxpi.evaluator import BaseEvaluator
from jaxpi.logging import Logger
from jaxpi.models import create_train_state
from jaxpi.utils import create_update_scheduler


def sample_batches(samplers):
    """Infinite batch iterator over a sampler or a dict of samplers.

    For a dict, each yielded batch is a dict with one entry drawn from every
    sampler (e.g. {"ics": ..., "res": ...}). Any iterable works as a sampler,
    so fixed batches can be supplied with itertools.repeat(batch).
    """
    if isinstance(samplers, dict):
        iterators = {key: iter(sampler) for key, sampler in samplers.items()}
        while True:
            yield {key: next(iterator) for key, iterator in iterators.items()}
    else:
        yield from iter(samplers)


def _ensure_wandb(config):
    """Initialize W&B once, on the first host only.

    The entity defaults to the logged-in account; set `config.wandb.entity`
    (e.g. your personal username) to log somewhere other than your default
    team, or run with WANDB_MODE=offline to keep runs local.
    """
    if jax.process_index() == 0 and wandb.run is None:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            entity=config.wandb.get("entity", None),
        )


def train_loop(
    config,
    model,
    batches,
    evaluator=None,
    logger=None,
    ckpt_mngr=None,
    eval_args=(),
    step_offset=0,
    stop_fn=None,
):
    """Run up to `config.training.max_steps` optimization steps on `model`.

    Args:
      batches: iterator of training batches (see `sample_batches`), or a
        sampler / dict of samplers, which is wrapped automatically.
      evaluator: metrics evaluator; defaults to `BaseEvaluator(config)`.
      logger: console logger; created automatically when omitted.
      ckpt_mngr: optional Orbax checkpoint manager; checkpoints are written
        every `config.saving.save_every_steps` steps and at the end. When
        `config.training.resume` is set and the manager already holds a
        checkpoint, training restores it and continues from its step.
      eval_args: extra positional arguments passed to the evaluator (e.g. the
        reference solution).
      step_offset: added to the step index when logging to W&B, so that
        consecutive windows/stages produce one continuous curve.
      stop_fn: optional `(step, log_dict) -> bool` early-stopping predicate,
        evaluated whenever metrics are logged.

    Returns the model (its `state` is updated in place).
    """
    if isinstance(batches, dict) or not hasattr(batches, "__next__"):
        batches = sample_batches(batches)
    evaluator = evaluator if evaluator is not None else BaseEvaluator(config)
    logger = logger if logger is not None else Logger()
    _ensure_wandb(config)

    loss_update = create_update_scheduler(**config.loss_weighting.update_schedule)
    pseudo_time_update = create_update_scheduler(**config.pseudo_time.update_schedule)

    # Resume from this run's checkpoint, if requested and available
    start_step = 0
    if (
        config.training.get("resume", False)
        and ckpt_mngr is not None
        and ckpt_mngr.latest_step() is not None
    ):
        model.state = restore_checkpoint(ckpt_mngr, model.state)
        start_step = int(model.state.step)
        logging.info("Resuming training from step {}".format(start_step))

    start_time = time.time()
    print("Waiting for JIT...")
    init_state = model.state  # Reference state for pseudo-time weight updates
    for step in range(start_step, config.training.max_steps):
        batch = next(batches)
        model.state, loss, loss_dict = model.step(model.state, batch)

        if config.pseudo_time.strategy == "dynamic" and pseudo_time_update(step):
            res_batch = batch["res"] if isinstance(batch, dict) else batch
            model.state = model.update_pts_weights(model.state, init_state, res_batch)

        if config.loss_weighting.strategy == "dynamic" and loss_update(step):
            model.state = model.update_loss_weights(model.state, batch)

        # Log training metrics; only host 0 records results
        should_stop = False
        if jax.process_index() == 0 and step % config.logging.log_every_steps == 0:
            end_time = time.time()
            log_dict = evaluator(model, model.state, loss_dict, batch, *eval_args)
            wandb.log(log_dict, step + step_offset)

            logger.log_iter(step, start_time, end_time, log_dict)
            start_time = time.time()

            should_stop = stop_fn is not None and stop_fn(step, log_dict)

        # Save checkpoint
        if ckpt_mngr is not None and config.saving.save_every_steps > 0:
            if (
                (step + 1) % config.saving.save_every_steps == 0
                or (step + 1) == config.training.max_steps
                or should_stop
            ):
                save_checkpoint(ckpt_mngr, model.state)

        if should_stop:
            logging.info("Stopping criterion met at step {}".format(step))
            break

    # Save final checkpoint
    if ckpt_mngr is not None and config.saving.save_every_steps > 0:
        save_checkpoint(ckpt_mngr, model.state)
        ckpt_mngr.wait_until_finished()

    return model


def train(config, model, batches, evaluator=None, eval_args=(), stop_fn=None):
    """Single-run training: `train_loop` with the standard checkpoint layout."""
    ckpt_mngr = create_checkpoint_manager(config.saving, get_ckpt_path(config))
    return train_loop(
        config,
        model,
        batches,
        evaluator=evaluator,
        ckpt_mngr=ckpt_mngr,
        eval_args=eval_args,
        stop_fn=stop_fn,
    )


def train_time_windows(
    config,
    model,
    make_samplers,
    evaluator=None,
    propagate_ic=None,
    make_eval_args=None,
):
    """Train a time-dependent problem forward over consecutive time windows.

    Args:
      make_samplers: `window_idx (0-based) -> sampler or dict of samplers` for
        that window.
      evaluator: metrics evaluator; defaults to `BaseEvaluator(config)`.
      propagate_ic: optional `(model, window_idx)`, invoked after each trained
        window (and after a resume restore) to turn the predicted solution at
        the end of the window into the next window's initial condition —
        typically by updating the arrays that `make_samplers` closes over. It
        may return a replacement model (e.g. rebuilt with the new IC when the
        IC is captured inside jitted closures); returning None keeps the
        current model.
      make_eval_args: optional `window_idx -> tuple` of extra positional
        arguments for the evaluator (e.g. the window's reference solution).

    Each window `idx` checkpoints under `time_window_{idx + 1}`.
    `config.training.num_time_windows` is the *total* number of windows. When
    `config.training.resume` is set, training continues where the checkpoints
    end: a partially trained window is re-entered and resumes from its last
    step, a finished window's successor starts fresh. When
    `config.training.transfer_learning` is set, each window starts from the
    previous window's parameters (with a fresh optimizer state); otherwise
    from a fresh initialization.
    """
    logger = Logger()
    _ensure_wandb(config)
    ckpt_path = get_ckpt_path(config)

    # Resume from the last checkpointed time window, if requested and available
    start_idx = 0
    if config.training.get("resume", False):
        latest = latest_time_window(ckpt_path)
        if latest > 0:
            ckpt_mngr = create_checkpoint_manager(
                config.saving, ckpt_path, suffix="time_window_{}".format(latest)
            )
            model.state = restore_checkpoint(ckpt_mngr, model.state)
            if int(model.state.step) < config.training.max_steps:
                # Window `latest` was interrupted mid-training: re-enter it
                # (train_loop restores its checkpoint and continues from its
                # step). Its IC must be regenerated from the *previous*
                # window's final parameters.
                start_idx = latest - 1
                logging.info(
                    "Resuming interrupted time window {} at step {}".format(
                        latest, int(model.state.step)
                    )
                )
                if start_idx > 0:
                    prev_mngr = create_checkpoint_manager(
                        config.saving,
                        ckpt_path,
                        suffix="time_window_{}".format(start_idx),
                    )
                    model.state = restore_checkpoint(prev_mngr, model.state)
                    if propagate_ic is not None:
                        model = propagate_ic(model, start_idx - 1) or model
            else:
                start_idx = latest
                logging.info(
                    "Time window {} complete, resuming with window {}".format(
                        latest, latest + 1
                    )
                )
                if propagate_ic is not None:
                    model = propagate_ic(model, start_idx - 1) or model

    for idx in range(start_idx, config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))

        if idx > 0:
            # Fresh optimizer state for each window; carry the parameters over
            # when transfer learning is enabled, otherwise re-initialize.
            params = model.state.params if config.training.transfer_learning else None
            model.state = create_train_state(model.config, model.tx, model.arch, params=params)

        batches = sample_batches(make_samplers(idx))
        eval_args = make_eval_args(idx) if make_eval_args is not None else ()
        ckpt_mngr = create_checkpoint_manager(
            config.saving, ckpt_path, suffix="time_window_{}".format(idx + 1)
        )

        model = train_loop(
            config,
            model,
            batches,
            evaluator=evaluator,
            logger=logger,
            ckpt_mngr=ckpt_mngr,
            eval_args=eval_args,
            step_offset=idx * config.training.max_steps,
        )

        if propagate_ic is not None:
            model = propagate_ic(model, idx) or model

    return model
