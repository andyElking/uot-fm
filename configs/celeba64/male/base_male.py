def get_male_config(config):
    # training
    config.training.num_steps = 400000
    config.training.eval_freq = 50000
    config.training.print_freq = 1000
    # data
    config.data.attribute_id = 20
    config.data.map_forward = True
    config.data.precomputed_stats_file = "celeba64_male"

    config.eval.labelwise = True

    return config
