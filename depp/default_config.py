default_config = \
{'exp_name': '',
'data_dir': '',
'summary_dir': 'summary',
'model_dir': 'model',
'distance_mode': 'L2',
'embedding_size': 512,
'sequence_length': 0,
'h_channel': 8,
'beta_ratio': 50000,

'tau_pres_end_epoch': 5,
'start_epoch': 0,
'epoch': 3001,
'batch_size': 32,
'cp': 1.0,
'resblock_num': 1,

'lr': 2e-4,
'lr_decay': 2000,
'backbone_tree_file': '',
'backbone_seq_file': '',
'query_seq_file': '',
'weighted_method': 'square_root_fm',
'distance_alpha': 1e7,
'dis_loss_ratio': 1.0,
'plot_freq': 1000,
'save_data_freq': 1000,
'save_model_freq': 3000,
'print_freq': 1000,
'stop_par': 200,
'gpus': [0],
'plot_all_freqdistance_depp.sh -q ./seq/query_seq/query_seq.fa -b ./seq/backbone_seq/backbone_seq.fa -m model/epoch\=599.ckpt -t tree/backbone_tree/backbone.newick -o dis_mat/': 2000,
'lr_update_freq': 100,
'bar_update_freq': 1,
'val_freq': 50,
'patience': 5,
'model_path': '',
'outdir': 'depp_distance',
'num_worker': 4,
'replicate_seq': False}

#def get_cfg_defaults():
#    """Get a yacs CfgNode object with default values for my_project."""
#    # Return a clone so that the defaults will not be altered
#    # This is for the "local variable" use pattern
#    return _C.clone()
