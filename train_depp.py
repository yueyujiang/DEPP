#!/usr/bin/python

import os
from depp import Model_pl
from depp import default_config
import pkg_resources

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger
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

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summary_dir = args.summary_dir

    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)

    model = Model_pl.model(args=args)

    logger = TensorBoardLogger(
        save_dir=args.summary_dir,
        name=args.exp_name
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=model_dir,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    print(model_dir)
    if args.gpus == 0:
        trainer = pl.Trainer(
            logger=logger,
            gpus=args.gpus,
            progress_bar_refresh_rate=args.bar_update_freq,
            check_val_every_n_epoch=args.val_freq,
            max_epochs=args.epoch,
            gradient_clip_val=args.cp,
            benchmark=True,
            callbacks=[early_stop_callback],
            checkpoint_callback=checkpoint_callback
        )
    else:
        trainer = pl.Trainer(
            logger=logger,
            gpus=args.gpus,
            progress_bar_refresh_rate=args.bar_update_freq,
            distributed_backend='ddp',
            check_val_every_n_epoch=args.val_freq,
            max_epochs=args.epoch,
            gradient_clip_val=args.cp,
            benchmark=True,
            callbacks=[early_stop_callback],
            checkpoint_callback=checkpoint_callback
        )

    trainer.fit(model)

if __name__ == '__main__':
    main()
