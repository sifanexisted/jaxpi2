import ml_collections

import jax.numpy as jnp


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "JAXPI-KF-Re1e6"
    wandb.name = "baseline"
    wandb.tag = None

    # Problem setup
    config.data_path = "data/kolmogorov_flow_Re1e6.npy"
    config.init_time_step = 0

    config.problem = problem = ml_collections.ConfigDict()
    problem.force_amplitude = 0.1  # body force: amplitude * sin(wavenumber * pi * y)
    problem.force_wavenumber = 4.0

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 2
    arch.hidden_dim = 512
    arch.out_dim = 3
    arch.activation = "swish"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (2 * jnp.pi, 2 * jnp.pi), "axis": (1, 2), "trainable": (False, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 2.0, "embed_dim": 512}
    )
    arch.nonlinearity = 1.0

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "soap"
    optim.lr_schedule = "exponential_decay"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.warmup_steps = 2000
    optim.staircase = False
    optim.schedule_free = True

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 50000
    training.batch_size = 8192
    training.ics_batch_size = 32768
    training.time_window_size = 0.1
    training.num_time_windows = 1
    training.transfer_learning = True
    # Continue from the last checkpointed time window on re-invocation
    training.resume = True

    # Global weightings for different loss terms
    config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
    loss_weighting.strategy = "dynamic"  # "dynamic" or "constant", constant means fixed weights
    loss_weighting.loss_weights = ml_collections.ConfigDict(
        {"u_ic": 100.0, "v_ic": 100.0, "ru": 1.0, "rv": 1.0, "rc": 1.0}
    )
    loss_weighting.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 1000,
    })
    loss_weighting.momentum = 0.9

    # Pseudo-time stepping for the PDE residuals
    config.pseudo_time = pseudo_time = ml_collections.ConfigDict()
    pseudo_time.enabled = False
    pseudo_time.strategy = "constant"  # "dynamic" or "constant", constant means fixed weights
    pseudo_time.pts_weights = ml_collections.ConfigDict(
        {"ru": 1.0, "rv": 1.0, "rc": 1.0})
    pseudo_time.update_schedule = ml_collections.ConfigDict({
        "start": 100,
        "every": 1000,
    })
    pseudo_time.momentum = 0.9
    pseudo_time.shrink = shrink = ml_collections.ConfigDict()
    shrink.enabled = True
    shrink.start_log_drop = 2.0
    shrink.end_log_drop = 6.0
    shrink.min_factor = 0.1

    config.causal = causal = ml_collections.ConfigDict()
    causal.enabled = True
    causal.num_chunks = 16
    causal.tol = 1.0

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_lr = True
    logging.log_losses = True
    logging.log_raw_losses = False
    logging.log_loss_weights = True
    logging.log_pts_weights = False
    logging.log_causal_weights = True
    logging.log_grads = False
    logging.log_nonlinearities = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.ckpt_path = None  # defaults to <cwd>/<wandb.name>/ckpt
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 2

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
