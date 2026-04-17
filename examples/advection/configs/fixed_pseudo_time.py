from configs.base import get_base_config


def get_config():
    config = get_base_config()
    config.wandb.name = "fixed_pseudo_time"

    config.pseudo_time.enabled = True
    config.pseudo_time.strategy = "constant"

    config.logging.log_pts_weights = True
    return config