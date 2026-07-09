import ml_collections


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "JAXPI-Taylor-Green"
    wandb.name = "baseline"
    wandb.tag = None

    # Problem setup
    config.Re = 1600.0
    config.grid_res = 256  # initial condition grid (analytic IC)
    config.ic_path = None  # optional .npy with a predicted flow field as IC

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 2
    arch.hidden_dim = 512
    arch.out_dim = 4
    arch.activation = "swish"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (1.0, 1.0, 1.0), "axis": (1, 2, 3), "trainable": (False, False, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 1.0, "embed_dim": 512}
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
    training.max_steps = 40000
    training.batch_size = 6144
    training.ics_batch_size = 24576
    training.time_window_size = 0.2
    training.num_time_windows = 1
    training.transfer_learning = True
    # Continue from the last checkpointed time window on re-invocation
    training.resume = True

    # Multi-stage training (used when mode == "train_multi_stage")
    config.multi_stage = multi_stage = ml_collections.ConfigDict()
    multi_stage.num_stages = 2
    multi_stage.eps_list = [1.0, 1e-3]
    multi_stage.freq_list = [1, 2]
    # Loss keys of the correction stages (stage index >= 1)
    multi_stage.loss_weights = ml_collections.ConfigDict(
        {"u0_diff": 100.0, "v0_diff": 100.0, "w0_diff": 100.0,
         "fu": 1.0, "fv": 1.0, "fw": 1.0, "fc": 1.0}
    )
    multi_stage.pts_weights = ml_collections.ConfigDict(
        {"fu": 1.0, "fv": 1.0, "fw": 1.0, "fc": 1.0}
    )
    multi_stage.rejection_sampling = rejection_sampling = ml_collections.ConfigDict()
    rejection_sampling.enabled = True
    rejection_sampling.num_samples = 2048
    rejection_sampling.threshold = 0.5
    rejection_sampling.batch_size = 4096
    multi_stage.early_stop = early_stop = ml_collections.ConfigDict()
    early_stop.enabled = False
    early_stop.start_step = 20000
    early_stop.rc_threshold_first = 1e-8
    early_stop.rc_threshold_later = 1e-12

    # Global weightings for different loss terms
    config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
    loss_weighting.strategy = "dynamic"  # "dynamic" or "constant", constant means fixed weights
    loss_weighting.loss_weights = ml_collections.ConfigDict(
        {"u_ic": 100.0, "v_ic": 100.0, "w_ic": 100.0,
         "ru": 1.0, "rv": 1.0, "rw": 1.0, "rc": 1.0}
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
        {"ru": 1.0, "rv": 1.0, "rw": 1.0, "rc": 1.0})
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
    logging.log_lr = True
    logging.log_losses = True
    logging.log_raw_losses = False
    logging.log_loss_weights = True
    logging.log_pts_weights = False
    logging.log_causal_weights = True
    logging.log_grads = False
    logging.log_nonlinearities = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.ckpt_path = None  # defaults to <cwd>/<wandb.name>/ckpt
    saving.save_every_steps = 5000
    saving.num_keep_ckpts = 2

    # Input shape for initializing Flax models
    config.input_dim = 4

    # Integer for PRNG random seed.
    config.seed = 43

    return config
