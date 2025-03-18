import copy

DEFAULTS = {
    "save_root": "/mnt/fs2/grantsrb/fca_saves/",
    "seed": 12345,
    "exp_name": "equal_distro",
    'task_params': {
        'n_pairs': 4,
        'n_samples': 1000,
        "operations": None,
    },
    
    'model_params': {
        'embedding_dim': 32,
        'd_model': 512,
        'n_layers': 3,
        'nonlinearity': "ReLU",
        'lnorm': True,
    },
    'lr': 0.001,
    'num_epochs': 1000000,
    'batch_size': 128,
    "patience": 100,
    "plateau": 0.005,
    'model_load_path': None,
    "save_to_load_path": True,

    "do_fca": False,
    "fine_tuning": False, # optionally bypass the initial accuracy threshold in fca runs
    "zscore_fca": False, # Will zscore the reps before fca but be careful
        # as this will make the resulting vectors non orthogonal in the
        # orig activation space
    'fca_load_path': None,
    "subtract_prev_fcas": False, # Will set previously learned FCA
        # components to remove_components=True. Irrelevant when
        # ensure_ortho_chain is True and they were trained with remove_components=False
    'fca_params': {
        "max_rank": None,
        "remove_components": False, # Can optionally learn components that
            # are subtracted from the reps instead of bottlenecking into them.
            # Mainly makes sense for fine tuning.
    },
    'fca_layers': ["hidden_layers.1"],
    "fca_acc_threshold": 0.99,
    "ensure_ortho_chain": True, # will ensure new FCAs are orthogonal to previous FCAs

    'persistent_keys': [
        'fca_params', 'fca_layers', 'lr',
        "fca_load_path", "batch_size", "num_epochs",
        "ensure_ortho_chain", "do_fca", "persistent_keys",
        "subtract_prev_fcas", "fca_save_folder",
        "fine_tuning",
    ],
}

CHAIN_DEFAULTS = {
    **copy.deepcopy(DEFAULTS),
    "patience": 200,
    "plateau": 0.001,

    "do_fca": True,
    'fca_layers': ["hidden_layers.1"],
    "fca_acc_threshold": 0.99, # stops early when fca module reaches this trn and val acc
    "ensure_ortho_chain": True, # will load the sd from the previous fca into the newest 
}
