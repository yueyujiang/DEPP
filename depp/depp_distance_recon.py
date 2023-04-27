#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
# import Model_pl
from depp import Agg_model_recon
from depp import Model_recon
from depp import utils
from depp import default_config
from omegaconf import OmegaConf

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def main():
    args_base = OmegaConf.create(default_config.default_config)

    args_cli = OmegaConf.from_cli()

    # if args_cli.config_file is not None:
    #     args_cfg = OmegaConf.load(args_cli.config_file)
    #     args_base = OmegaConf.merge(args_base, args_cfg)
    #     args_base.exp_name = os.path.splitext(os.path.basename(args_cli.config_file))[0]
    # elif args_cli.exp_name is None:
    #     raise ValueError('exp_name cannot be empty without specifying a config file')
    # del args_cli['config_file']
    args = OmegaConf.merge(args_base, args_cli)
    cluster_model = True
    try:
        if not torch.cuda.is_available():
            m = torch.load(args.model_path, map_location=torch.device('cpu'))
        else:
            m = torch.load(args.model_path)
        classifier_cluster_num = m['state_dict']['classifier.linear.bias'].shape[0]
        model = Agg_model_recon.model.load_from_checkpoint(args.model_path, load_model=False, classifier_cluster_num=classifier_cluster_num)
    except:
        cluster_model = False
        try:
            model = Model_recon.model.load_from_checkpoint(args.model_path, current_model=0)
        except:
            print(f'cannot load model {args.model_path}')
            sys.exit()
    # if args.recon_model_path:
    #     recon_model = Model_recon.model.load_from_checkpoint(args.recon_model_path)
    #     recon_model.is_training=False
    # else:
    #     recon_model = None
    if cluster_model:
        utils.save_depp_dist_cluster(model, args, use_cluster=args.use_cluster, recon_model=model)
    else:
        utils.save_depp_dist(model, args)

    if args.get_representative_emb:
        utils.save_repr_emb(model, args)

if __name__ == '__main__':
    main()

