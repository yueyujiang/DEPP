#!/home/y5jiang/miniconda3/envs/depp_env/bin/python

import os
from depp import Model_pl
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

    if args.recon_model_path:
        if not os.path.isfile(args.recon_model_path):
            print(f"{args.recon_model_path} not exist.")
            recon_model = None
        else:
            recon_model = Model_recon.model.load_from_checkpoint(args.recon_model_path)
            recon_model.is_training=False
    else:
        recon_model = None

    if args.model_path:
        if not os.path.isfile(args.model_path):
            print(f"{args.model_path} not exist.")
            model = None
        else:
            model = Model_pl.model.load_from_checkpoint(args.model_path)
    else:
        model = None

    if model is None and recon_model is None:
        print('No model exist. exit...')
        return

    utils.save_depp_dist(model=model, args=args, recon_model=recon_model)

if __name__ == '__main__':
    main()

