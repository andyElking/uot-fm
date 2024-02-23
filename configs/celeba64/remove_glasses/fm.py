from configs.base_fm import get_fm_config
from configs.celeba64.base_unet import get_unet_config
from configs.celeba64.base_celeba import get_celeba_config
from configs.celeba64.remove_glasses.base_glasses import get_glasses_config


def get_config():
    config = get_fm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_glasses_config(config)
    config.name = "fm_celeba64_remove-glasses"

    return config
