#!/usr/bin/env python3

import torch
import os
import math
import torch.nn as nn
import submodule
import data_recon
import utils
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
#
#
# class SeqNet(nn.Module):
#     def __init__(self, args):
#         super(SeqNet, self).__init__()
#         self.conv1 = nn.Conv1d(4+1, 8, 1)
#         self.lstm = nn.LSTM(8, 8, 2, batch_first=True, bidirectional=True)
#         self.conv2 = nn.Conv1d(16, 4, 1)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         n, chn, l = x.shape
#         # idx = torch.zeros(n, l, l).to(x.device)
#         idx = torch.arange(0, l).to(x.device).view(1, 1, -1).repeat(len(x), 1, 1) * 0.01
#         # idx[:, torch.arange(0, l), torch.arange(0, l)] = 1
#         x = torch.cat([x, idx], dim=1)
#         x = self.relu(self.conv1(x))
#         x = self.lstm(x.permute(0, 2, 1))
#         x = self.conv2(x[0].permute(0, 2, 1))
#         return x

class SeqNet(nn.Module):
    def __init__(self, args):
        super(SeqNet, self).__init__()
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
        args.seq_h = int(args.sequence_length * args.seq_h_ratio)
        self.linear = nn.Conv1d(args.h_channel,
                                args.seq_h,
                                args.sequence_length)
        self.tranconv = nn.ConvTranspose1d(args.seq_h, 4, args.sequence_length)
        self.args = args
        self.train_loss = 0
        print("seq_h", args.seq_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, channel, seq_length = x.shape

        x = self.celu(self.conv(x))
        x = self.resblocks(x)
        x = self.celu(self.linear(x))
        x = self.tranconv(x)
        return x

# class SeqNet(nn.Module):
#     def __init__(self, args):
#         super(SeqNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(5, 16, 5, stride=2),
#             nn.ReLU(),
#             nn.Conv1d(16, 32, 5, stride=2),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, 5, stride=2),
#             nn.ReLU(),
#             nn.Conv1d(64, 128, 5, stride=2),
#             nn.ReLU(),
#             nn.Conv1d(128, 128, 5, stride=2),
#             nn.ReLU()
#         )
#         self.lstm = nn.LSTM(128, 128, 1, batch_first=True, bidirectional=True)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(128*2, 128, 5, stride=2, padding=0, output_padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose1d(128, 64, 5, stride=2, padding=0, output_padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 32, 5, stride=2, padding=0, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 16, 5, stride=2, padding=0, output_padding=0),
#             nn.ReLU(),
#             nn.ConvTranspose1d(16, 16, 5, stride=2, padding=0, output_padding=0),
#             nn.ReLU(),
#             nn.Conv1d(16, 4, 1)
#         )
#         self.L = args.sequence_length
#
#     def forward(self, x):
#         idx = torch.arange(0, self.L).to(x.device).view(1, 1, -1).repeat(len(x), 1, 1) * 0.01
#                 # idx[:, torch.arange(0, l), torch.arange(0, l)] = 1
#         x = torch.cat([x, idx], dim=1)
#         x = self.encoder(x)
#         x = self.lstm(x.permute(0, 2, 1))
#         x = self.decoder(x[0].permute(0, 2, 1))
#         # x = self.decoder(x)
#         if self.L % 2 == 0:
#             x = x[:, :, :-1]
#         return x

class model(LightningModule):
    def __init__(self, args):
        super(model, self).__init__()
        self.save_hyperparameters(args)
        if not self.hparams.sequence_length:
            utils.get_seq_length(self.hparams)
        self.save_hyperparameters(self.hparams)
        self.encoder = encoder(self.hparams)
        self.channel = 4
        self.hparams.distance_ratio = math.sqrt(
            float(1.0 / float(self.hparams.embedding_size) / 10 * float(self.hparams.distance_alpha)))

        self.dis_loss_w = 100
        self.train_loss = []
        self.val_loss = float('inf')
        self.seq_net = SeqNet(self.hparams)
        self.running_recon_loss = 0
        self.reconseq = []
        self.wrong_site = []
        self.is_training = True
        self.cnt = []

    def forward(self, x: torch.Tensor, train_sn=False, mask=None) -> torch.Tensor:
        recon_x = self.seq_net(x)
        softmax_recon_x = torch.softmax(recon_x, dim=1)
        if not train_sn:
            softmax_recon_x[mask.repeat(1, 4, 1)] = x[mask.repeat(1, 4, 1)]
        encoding = self.encoder(softmax_recon_x)
        if self.is_training:
            return recon_x, encoding
        else:
            return encoding

    def training_step(self, batch, batch_idx):
        nodes = batch['nodes']
        seq = batch['seqs'].float()
        masked_seq = batch['masked_seqs'].float()
        mask = batch['mask']
        device = seq.device

        if self.trainer.current_epoch < self.hparams.ae_train_epoch:
            reconseq, _ = self(masked_seq, train_sn=True)
            n, chn, l = reconseq.shape
            reconseq = reconseq.permute(0, 2, 1).reshape(n*l, chn)
            mask = mask.permute(0, 2, 1).reshape(n*l, -1).squeeze(-1)
            seq = seq.permute(0, 2, 1).reshape(n*l, chn)
            loss = nn.functional.cross_entropy(reconseq[mask, :], torch.argmax(seq[mask, :], dim=-1))
            self.running_recon_loss += loss
            if self.trainer.current_epoch % 10 == 0:
                self.reconseq.append(reconseq.flatten())
                wrong_site = (torch.argmax(reconseq[mask, :], dim=-1) != torch.argmax(seq[mask, :], dim=-1)).sum() / mask.sum()
                self.wrong_site.append(wrong_site)
        else:
            self.train_data.train_recon = False
            _, encoding = self(masked_seq, mask=mask)
            gt_distance = self.train_data.true_distance(nodes, nodes).to(device)

            distance = utils.distance(encoding, encoding.detach(),
                                      self.hparams.distance_mode) * self.hparams.distance_ratio

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
        if self.trainer.current_epoch % 100 == 0:
            self.trainer.save_checkpoint(f'{self.hparams.model_dir}/epoch-{self.trainer.current_epoch}.pth')
            #if self.trainer.current_epoch > 0:
            #    os.remove(f'{self.hparams.model_dir}/epoch-{self.trainer.current_epoch - 100}.pth')
    def on_train_epoch_start(self):
        if self.trainer.current_epoch == self.hparams.ae_train_epoch:
            self.seq_net.requires_grad_(False)
            self.train_data.train_encoder = False


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
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch < self.hparams.ae_train_epoch:
            # self.logger.experiment.add_scalar('recon_loss', self.running_recon_loss, self.trainer.current_epoch)
            # self.logger.experiment.add_histogram('recon_seq', torch.sigmoid(torch.cat(self.reconseq)), self.trainer.current_epoch)
            # self.logger.experiment.add_histogram('wrong_site', torch.tensor(self.wrong_site), self.trainer.current_epoch)
            self.running_recon_loss = 0
            self.reconseq = []
            self.wrong_site = []
        self.log('val_loss', val_loss)
        self.val_loss = 0

    def val_dataloader(self):
        # TODO: do a real train/val split
        self.train_data = data_recon.data(self.hparams, calculate_distance_matrix=True)
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

            #lr = 3e-5 + self.hparams.lr * (0.1 ** (epoch / self.hparams.lr_decay))
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
