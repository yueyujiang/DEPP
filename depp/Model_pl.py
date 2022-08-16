#!/usr/bin/env python3

import torch
import os
import math
import torch.nn as nn
from depp import submodule
from depp import data
from depp import utils
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable, Union
from torch.optim.optimizer import Optimizer
import math


class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        channel = 4

        self.conv = nn.Conv1d(channel, args.h_channel, 1)
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        resblocks = []
        for i in range(args.resblock_num):
            resblocks.append(submodule.resblock(args.h_channel,
                                                args.h_channel,
                                                5, 0.3))
        self.resblocks = nn.Sequential(*resblocks)

        self.linear = nn.Conv1d(args.h_channel,
                                args.embedding_size,
                                args.sequence_length)
        self.args = args
        self.train_loss = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, channel, seq_length = x.shape

        x = self.celu(self.conv(x))
        x = self.resblocks(x)
        x = self.linear(x).squeeze(-1)
        x = x.view(bs, self.args.embedding_size)
        return x


class model(LightningModule):
    def __init__(self, args):
        super(model, self).__init__()
        self.save_hyperparameters(args)
        if not self.hparams.sequence_length:
            utils.get_seq_length(self.hparams)
        self.save_hyperparameters(self.hparams)
        self.encoder = encoder(self.hparams)
        self.channel = 4
        self.hparams.distance_ratio = math.sqrt(float(1.0 / float(self.hparams.embedding_size) / 10 * float(self.hparams.distance_alpha)))

        self.dis_loss_w = 100
        self.train_loss = []
        self.val_loss = float('inf') 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(x)
        return encoding

    def training_step(self, batch, batch_idx):

        nodes = batch['nodes']
        seq = batch['seqs'].float()
        device = seq.device

        encoding = self(seq)
        gt_distance = self.train_data.true_distance(nodes, nodes).to(device)

        distance = utils.distance(encoding, encoding.detach(), self.hparams.distance_mode) * self.hparams.distance_ratio

        not_self = torch.ones_like(distance)
        not_self[torch.arange(0, len(distance)), torch.arange(0, len(distance))] = 0

        dis_loss = utils.mse_loss(distance[not_self == 1], gt_distance[not_self == 1], self.hparams.weighted_method)
        loss = self.dis_loss_w * dis_loss * self.hparams.dis_loss_ratio
        self.val_loss += loss

        return {'loss': loss}

    def training_epoch_end(
            self,
            outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        self.dis_loss_w = 100 + 1e-3 * (self.trainer.current_epoch - 1e4) * (self.trainer.current_epoch > 1e4)
        if self.trainer.current_epoch % 1000 == 0:
            self.trainer.save_checkpoint(f'{self.hparams.model_dir}/epoch-{self.trainer.current_epoch}.pth')
            if self.trainer.current_epoch > 0:
                try:
                    os.remove(f'{self.hparams.model_dir}/epoch-{self.trainer.current_epoch - 100}.pth')
                except:
                    pass

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_data,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_worker,
                            shuffle=True,
                            drop_last=True)
        return loader

    def validation_step(self, batch, batch_idx):
        return {}        

    def validation_epoch_end(self, outputs):
        val_loss = self.val_loss
        self.val_loss = 0
        self.log('val_loss', val_loss)

    def val_dataloader(self):
        # TODO: do a real train/val split
        self.train_data = data.data_mask(self.hparams, calculate_distance_matrix=True)
        loader = DataLoader(self.train_data,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=self.hparams.num_worker,
                            drop_last=False)
        return loader

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:

        if self.trainer.global_step % self.hparams.lr_update_freq == 0:

            lr = 3e-5 + self.hparams.lr * (0.1 ** (epoch / self.hparams.lr_decay))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        super(model, self).optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu,
            using_native_amp,
            using_lbfgs,
        )
