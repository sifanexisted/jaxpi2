from configs.base import get_base_config


def get_config():
    config = get_base_config()

    config.wandb.name = "pseudo_time"

    config.pseudo_time.enabled = True
    config.pseudo_time.strategy = "dynamic"
    config.pseudo_time.update_schedule.every = 5000


    return config