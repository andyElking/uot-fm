import ml_collections


def get_base_config():
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.overfit_to_one_batch = False
    config.wandb_key = ""
    config.wandb_group = ""
    config.wandb_entity = ""

    # training
    config.training = training = ml_collections.ConfigDict()
    training.print_freq = 1000
    training.save_checkpoints = True
    training.preemption_ckpt = False
    training.ckpt_freq = 10000
    training.resume_ckpt = False

    config.eval = eval = ml_collections.ConfigDict()
    eval.compute_metrics = True
    eval.enable_fid = True
    eval.enable_path_lengths = True
    eval.enable_mse = False
    eval.checkpoint_metric = "fid"
    eval.save_samples = True
    eval.num_save_samples = 7
    eval.labelwise = True
    eval.checkpoint_step = 250000
    eval.name = ""

    eval.pid = pid = ml_collections.ConfigDict()
    pid.atol = 1e-3
    pid.rtol = 0.0
    pid.pcoeff = 0.0
    pid.icoeff = 1.0

    config.noisy = noisy = ml_collections.ConfigDict()
    noisy.enable = True
    noisy.t = 1.0
    noisy.s = 0.9
    noisy.tol_ratio = 0.5
    noisy.alpha = 1.0

    return config
