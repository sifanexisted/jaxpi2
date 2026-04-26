import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "JAXPI-advection"
    wandb.name = "plain"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "mlp"
    arch.num_layers = 3
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (1.0,), "axis": (1,), "trainable": (False,)}
    )
    arch.fourier_emb = ml_collections.ConfigDict(
        {"embed_scale": 1.0, "embed_dim": 256}
    )
    # arch.nonlinearity = 0.0

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "adam"
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
    training.max_steps = 100000
    training.batch_size = 1024

    # Global weightings for different loss terms
    config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
    loss_weighting.strategy = "constant"  # "dynamic" or "constant", constant means fixed weights
    loss_weighting.loss_weights = ml_collections.ConfigDict(
        {
            "ics": 1.0,
            "res": 1.0,
        }
    )
    loss_weighting.update_schedule = ml_collections.ConfigDict({
        "start": 10,
        "every": 1000,  # used when schedule="constant"
    })
    loss_weighting.momentum = 0.9

    # Pseudo-time stepping for the PDE residuals
    config.pseudo_time = pseudo_time = ml_collections.ConfigDict()
    pseudo_time.enabled = False
    pseudo_time.strategy = "dynamic"  # "dynamic" or "constant" constant means fixed weights
    pseudo_time.pts_weights = ml_collections.ConfigDict(
        {"res": 1.0})
    pseudo_time.update_schedule = ml_collections.ConfigDict({
        "start": 10,
        "every": 100,  # used when schedule="constant"
    })
    pseudo_time.momentum = 0.9
    pseudo_time.shrink = shrink = ml_collections.ConfigDict()
    shrink.enabled = True
    shrink.start_log_drop = 2.0
    shrink.end_log_drop = 6.0
    shrink.min_factor = 0.1

    config.causal = causal = ml_collections.ConfigDict()
    causal.enabled = False
    causal.num_chunks = 16
    causal.tol = 1.0

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_lr = True
    logging.log_losses = True
    logging.log_loss_weights = True
    logging.log_pts_weights = True
    logging.log_causal_weights = True
    logging.log_grads = False
    logging.log_nonlinearities = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 2

    # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config
