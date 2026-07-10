from absl import logging

import ml_collections

import jax
from jax import random
import jax.numpy as jnp

from jaxpi.models import create_model, create_train_state
from jaxpi.samplers import UniformSampler, MeshSampler
from jaxpi.checkpointing import create_checkpoint_manager, get_ckpt_path, latest_time_window, restore_checkpoint
from jaxpi.training import sample_batches, train_loop
from jaxpi.utils import get_eval_params

import models
from evaluators import NavierStokes3DEvaluator
from train import get_initial_condition, predict_initial_condition
from utils import reject_sampling


def make_stage_config(config, stage_idx):
    """Per-stage config: later stages train a correction network with its own
    loss keys, a larger Fourier embedding scale, and no causal weighting."""
    stage_config = ml_collections.ConfigDict(config.to_dict())
    if stage_idx > 0:
        stage_config.arch.fourier_emb.embed_scale = float(config.multi_stage.freq_list[stage_idx])
        stage_config.loss_weighting.loss_weights = dict(config.multi_stage.loss_weights)
        stage_config.pseudo_time.pts_weights = dict(config.multi_stage.pts_weights)
        stage_config.causal.enabled = False
    return stage_config


def create_stage_model(config, stage_idx, t_max, nu, prev_params_list):
    """Build the model for a stage: the base PINN for stage 0, a MultiStage
    correction model (around the frozen previous stages) otherwise."""
    stage_config = make_stage_config(config, stage_idx)

    if stage_idx == 0:
        return create_model(stage_config, models.NavierStokes3D, t_max=t_max, nu=nu)

    eps_list = list(config.multi_stage.eps_list[: stage_idx + 1])
    return create_model(
        stage_config, models.MultiStage, t_max=t_max, nu=nu,
        prev_params_list=prev_params_list, eps_list=eps_list,
    )


def sample_residual_points(config, model, dom, stage_idx):
    """Extra collocation points concentrated where the frozen previous-stage
    residual is large, via rejection sampling."""
    rs = config.multi_stage.rejection_sampling
    if not rs.enabled:
        return None

    assert rs.num_samples % jax.device_count() == 0, (
        "multi_stage.rejection_sampling.num_samples must be divisible by the "
        "number of devices"
    )

    def residual_magnitude(points):
        ru, rv, rw, rc = model.r_prev_pred_fn(
            points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        )
        return jnp.sqrt(ru**2 + rv**2 + rw**2 + rc**2)

    key = random.PRNGKey(config.seed + 300 + stage_idx)
    samples = reject_sampling(
        key, rs.num_samples, residual_magnitude, dom,
        threshold=rs.threshold, batch_size=rs.batch_size,
    )
    return samples


def train_stage(config, model, evaluator, samplers, window_idx, stage_idx, extra_samples=None):
    num_stages = config.multi_stage.num_stages

    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(
        config.saving, get_ckpt_path(config),
        suffix="time_window_{}_stage_{}".format(window_idx, stage_idx + 1),
    )

    # Batch iterator; append the rejection-sampled points to every residual batch
    batches = sample_batches(samplers)
    if extra_samples is not None:
        def with_extra_samples(base=batches):
            for batch in base:
                batch["res"] = jnp.concatenate([batch["res"], extra_samples], axis=0)
                yield batch
        batches = with_extra_samples()

    # Optionally stop the stage early once the continuity residual of the
    # (composed) solution is converged
    early_stop = config.multi_stage.early_stop
    stop_fn = None
    if early_stop.enabled:
        rc_threshold = (
            early_stop.rc_threshold_first if stage_idx == 0 else early_stop.rc_threshold_later
        )

        def stop_fn(step, log_dict):
            return step >= early_stop.start_step and log_dict["loss/p_res"] < rc_threshold

    step_offset = ((window_idx - 1) * num_stages + stage_idx) * config.training.max_steps
    return train_loop(
        config, model, batches, evaluator=evaluator, ckpt_mngr=ckpt_mngr,
        step_offset=step_offset, stop_fn=stop_fn,
    )


def train_one_window(config, evaluator, u0, v0, w0, coords, dom, t_max, nu, window_idx, transfer=False):
    """Train all stages of one time window. Returns the final stage model."""
    num_stages = config.multi_stage.num_stages

    prev_params_list = []
    model = None
    for stage_idx in range(num_stages):
        if stage_idx > 0:
            logging.info(
                "Training stage {}, eps: {}, Fourier frequency: {}".format(
                    stage_idx + 1,
                    config.multi_stage.eps_list[stage_idx],
                    config.multi_stage.freq_list[stage_idx],
                )
            )

        model = create_stage_model(config, stage_idx, t_max, nu, prev_params_list)

        # Transfer learning: initialize from the same stage of the previous window
        if transfer:
            ckpt_mngr = create_checkpoint_manager(
                config.saving, get_ckpt_path(config),
                suffix="time_window_{}_stage_{}".format(window_idx - 1, stage_idx + 1),
            )
            state = restore_checkpoint(ckpt_mngr, model.state)
            model.state = create_train_state(
                model.config, tx=model.tx, arch=model.arch, params=state.params
            )
            logging.info(
                "Transferred stage {} parameters from time window {}".format(
                    stage_idx + 1, window_idx - 1
                )
            )

        # Extra collocation points where the previous stages' residual is large
        extra_samples = None
        if stage_idx > 0:
            extra_samples = sample_residual_points(config, model, dom, stage_idx)

        # Initialize the samplers; fold window and stage indices into the RNG
        # keys so every stage trains on a different collocation sequence
        uvw_0 = jnp.stack([u0, v0, w0], axis=-1)
        ics_sampler = MeshSampler(
            coords, uvw_0, config.training.ics_batch_size,
            rng_key=random.PRNGKey(config.seed + 100 + window_idx * num_stages + stage_idx),
        )
        res_sampler = UniformSampler(
            dom, config.training.batch_size,
            rng_key=random.PRNGKey(config.seed + 200 + window_idx * num_stages + stage_idx),
        )

        samplers = {
            "ics": iter(ics_sampler),
            "res": iter(res_sampler),
        }

        model = train_stage(
            config, model, evaluator, samplers, window_idx, stage_idx,
            extra_samples=extra_samples,
        )

        # Freeze this stage for the next one. Use the schedule-free averaged
        # parameters, which are the ones that define the stage's solution.
        prev_params_list.append(get_eval_params(model.state, config.optim.schedule_free))

    return model


def restore_window_model(config, t_max, nu, window_idx):
    """Rebuild the final multi-stage model of a fully trained time window."""
    num_stages = config.multi_stage.num_stages
    ckpt_path = get_ckpt_path(config)

    prev_params_list = []
    model = None
    for stage_idx in range(num_stages):
        model = create_stage_model(config, stage_idx, t_max, nu, prev_params_list)
        ckpt_mngr = create_checkpoint_manager(
            config.saving, ckpt_path,
            suffix="time_window_{}_stage_{}".format(window_idx, stage_idx + 1),
        )
        model.state = restore_checkpoint(ckpt_mngr, model.state)
        logging.info(
            "Restored time window {} stage {} at step {}".format(
                window_idx, stage_idx + 1, int(model.state.step)
            )
        )
        prev_params_list.append(get_eval_params(model.state, config.optim.schedule_free))

    return model


def train_and_evaluate(config, workdir):
    u0, v0, w0, coords = get_initial_condition(config)
    nu = 1.0 / config.Re

    # Define the time and space domain of one time window
    dT = config.training.time_window_size
    t0 = 0.0
    t1 = 1.05 * dT

    dom = jnp.array([
        [t0, t1],
        [0.0, 2.0 * jnp.pi],
        [0.0, 2.0 * jnp.pi],
        [0.0, 2.0 * jnp.pi],
    ])

    # Initialize evaluator
    evaluator = NavierStokes3DEvaluator(config)

    # Resume from the last fully trained time window, if any: restore all of
    # its stages and use the composed solution at t = dT as the next initial
    # condition.
    start_idx = 0
    if config.training.resume:
        ckpt_path = get_ckpt_path(config)
        # A window only counts as trained once its final stage is checkpointed
        start_idx = latest_time_window(
            ckpt_path,
            pattern=r"time_window_(\d+)_stage_{}".format(config.multi_stage.num_stages),
        )
        if start_idx > 0:
            restored_model = restore_window_model(config, dT, nu, start_idx)
            u0, v0, w0, enstrophy = predict_initial_condition(config, restored_model, dT, coords)
            logging.info(
                "Resuming from time window {}, enstrophy at t = {:.3f}: {:.5f}".format(
                    start_idx, start_idx * dT, enstrophy
                )
            )

    for idx in range(start_idx, start_idx + config.training.num_time_windows):
        logging.info(
            "Training time window {}, start time: {:.3f}, end time: {:.3f}".format(
                idx + 1, idx * dT, (idx + 1) * dT
            )
        )

        transfer = config.training.transfer_learning and idx > 0

        model = train_one_window(
            config, evaluator, u0, v0, w0, coords, dom,
            t_max=dT, nu=nu, window_idx=idx + 1, transfer=transfer,
        )

        # Update the initial condition for the next time window
        u0, v0, w0, enstrophy = predict_initial_condition(config, model, dT, coords)
        logging.info(
            "Time window {} done, enstrophy at t = {:.3f}: {:.5f}".format(
                idx + 1, (idx + 1) * dT, enstrophy
            )
        )
