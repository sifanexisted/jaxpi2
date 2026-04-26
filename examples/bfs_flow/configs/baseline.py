from configs.base import get_base_config

def get_config():
    config = get_base_config()
    config.wandb.name = "baseline"
    return config