from configs.base_otfm import get_otfm_config
from configs.celeba256.base_unet import get_unet_config
from configs.celeba256.base_celeba import get_celeba_config
from configs.celeba256.male.base_male import get_male_config


def get_config():
    config = get_otfm_config()
    config = get_unet_config(config)
    config = get_celeba_config(config)
    config = get_male_config(config)
    config.name = "ot-fm_celeba256_male"

    return config
