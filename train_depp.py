#!/usr/bin/env python3

import os
from depp import Model_pl
from depp import default_config
import pkg_resources
from depp import Agg_model

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf


# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

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
    os.makedirs(model_dir, exist_ok=True)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    print(model_dir)

    if args.backbone_tree_file is not None and args.cluster_num == 1:
        args.start_model_idx = 0
        args.end_model_idx = 1
    # trainer.fit(model)
    if args.end_model_idx is None:
        args.end_model_idx = args.cluster_num
    if args.start_model_idx == -1:
        if args.gpus == 0:
            classifier_trainer = pl.Trainer(
                logger=False,
                gpus=args.gpus,
                # progress_bar_refresh_rate=args.bar_update_freq,
                check_val_every_n_epoch=args.val_freq,
                max_epochs=args.classifier_epoch + 1,
                gradient_clip_val=args.cp,
                benchmark=True,
                callbacks=[early_stop_callback],
                enable_checkpointing=checkpoint_callback
            )
        else:
            classifier_trainer = pl.Trainer(
                logger=False,
                gpus=args.gpus,
                # progress_bar_refresh_rate=args.bar_update_freq,
                strategy='ddp',
                check_val_every_n_epoch=args.val_freq,
                max_epochs=args.classifier_epoch + 1,
                gradient_clip_val=args.cp,
                benchmark=True,
                callbacks=[early_stop_callback],
                enable_checkpointing=checkpoint_callback
            )
        model = Model_pl.model(args=args, current_model=-1)
        classifier_trainer.fit(model)
        args.start_model_idx = 0

    for model_idx in range(args.start_model_idx, args.end_model_idx):
        if args.gpus == 0:
            trainer = pl.Trainer(
                logger=False,
                gpus=args.gpus,
                # progress_bar_refresh_rate=args.bar_update_freq,
                check_val_every_n_epoch=args.val_freq,
                max_epochs=args.epoch + 1,
                gradient_clip_val=args.cp,
                benchmark=True,
                callbacks=[early_stop_callback],
                enable_checkpointing=checkpoint_callback
            )
        else:
            trainer = pl.Trainer(
                logger=False,
                gpus=args.gpus,
                # progress_bar_refresh_rate=args.bar_update_freq,
                strategy='ddp',
                check_val_every_n_epoch=args.val_freq,
                max_epochs=args.epoch + 1,
                gradient_clip_val=args.cp,
                benchmark=True,
                callbacks=[early_stop_callback],
                enable_checkpointing=checkpoint_callback
            )

        model = Model_pl.model(args=args, current_model=model_idx)
        trainer.fit(model)

    if args.backbone_tree_file is not None and args.cluster_num == 1:
        os.rename(f'{args.model_dir}/0/epoch-{args.epoch - args.epoch % 100}.pth', f'{args.model_dir}/depp-model.pth')
    else:
        model = Agg_model.model(args=args, load_model=True)
        trainer = pl.Trainer(
            logger=False,
            gpus=args.gpus,
            progress_bar_refresh_rate=args.bar_update_freq,
            # distributed_backend='ddp',
            check_val_every_n_epoch=args.val_freq,
            max_epochs=1,
            gradient_clip_val=args.cp,
            benchmark=True
            # reload_dataloaders_every_epoch=True
        )
        trainer.fit(model)


if __name__ == '__main__':
    main()
