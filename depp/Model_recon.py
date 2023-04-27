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

        self.seq_net = SeqNet(args)
        for param in self.seq_net.parameters():
            param.requires_grad = False

        self.args = args
        self.train_loss = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, channel, seq_length = x.shape

        mask = (x == self.args.gap_encode).all(1).unsqueeze(1).repeat(1, 4, 1)
        recon_x = self.seq_net(x)
        softmax_recon_x = torch.softmax(recon_x, dim=1)
        sorted_softmax = softmax_recon_x.sort(dim=1)[0]
        valid_sites = ((sorted_softmax[:, -1] / sorted_softmax[:, -2]) > 20).unsqueeze(1).repeat(1, 4, 1)
        x = x.clone()
        x[mask & valid_sites] = softmax_recon_x[mask & valid_sites]

        x = self.celu(self.conv(x))
        x = self.resblocks(x)
        x = self.linear(x).squeeze(-1)
        x = x.view(bs, -1)
        return x

class classifier(nn.Module):
    def __init__(self, args, cluster_num=None):
        super(classifier, self).__init__()
        channel = 4

        self.conv = nn.Conv1d(channel, args.h_channel, 1)
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        resblocks = []
        for i in range(args.resblock_num):
            resblocks.append(submodule.resblock(args.h_channel,
                                                args.h_channel,
                                                5, 0.3))

        if args.classifier_seqdir is not None:
            seqdir = args.classifier_seqdir
        else:
            seqdir = args.seqdir
        if cluster_num is None:
            cluster_num = len([i for i in range(len(os.listdir(seqdir))) if os.path.isfile(f'{seqdir}/{i}.fa')])

        self.resblocks = nn.Sequential(*resblocks)

        self.linear = nn.Conv1d(args.h_channel,
                                cluster_num,
                                args.sequence_length)

        self.seq_net = SeqNet(args)
        for param in self.seq_net.parameters():
            param.requires_grad = False

        self.args = args
        self.train_loss = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, channel, seq_length = x.shape

        mask = (x == self.args.gap_encode).all(1).unsqueeze(1).repeat(1, 4, 1)
        recon_x = self.seq_net(x)
        softmax_recon_x = torch.softmax(recon_x, dim=1)
        sorted_softmax = softmax_recon_x.sort(dim=1)[0]
        valid_sites = ((sorted_softmax[:, -1] / sorted_softmax[:, -2]) > 20).unsqueeze(1).repeat(1, 4, 1)
        x = x.clone()
        x[mask & valid_sites] = softmax_recon_x[mask & valid_sites]

        x = self.celu(self.conv(x))
        x = self.resblocks(x)
        x = self.linear(x).squeeze(-1)
        x = x.view(bs, -1)
        return x

class model(LightningModule):
    def __init__(self, args, current_model):
        super(model, self).__init__()
        self.save_hyperparameters(args)
        if not self.hparams.sequence_length:
            utils.get_seq_length(self.hparams)

        if current_model != -1:
            embedding_size = self.hparams.embedding_size
            self.hparams.embedding_size = embedding_size[current_model]
            self.encoder = encoder(self.hparams)
            self.hparams.embedding_size = embedding_size
        else:
            self.classifier = classifier(self.hparams)
        self.channel = 4
        self.hparams.distance_ratio = math.sqrt(float(1.0 / 128 / 10 * float(self.hparams.distance_alpha)))

        self.dis_loss_w = 100
        self.train_loss = []
        self.val_loss = float('inf')

        self.testing = False
        if current_model == -1:
            self.training_classifier = True
        else:
            self.training_classifier = False
        self.current_model = current_model
        self.counting = 0

        self.save_hyperparameters(self.hparams)

    def forward(self, x, model_idx=None) -> torch.Tensor:
        if self.training_classifier:
            return self.classifier(x)
        return self.encoder(x)

    def training_step(self, batch, batch_idx):

        nodes = batch['nodes']
        seq = batch['seqs'].float()
        model_idx = batch['cluster_idx'].long()
        device = seq.device

        if self.training_classifier:
            c = self(seq)
            loss = nn.functional.cross_entropy(c, model_idx)
            self.val_loss += loss
            return {'loss': loss}
        else:
            encoding = self(seq, model_idx[0])
            gt_distance = self.train_data.true_distance(nodes, nodes).to(device)

            distance = utils.distance(encoding, encoding.detach(), self.hparams.distance_mode) * self.hparams.distance_ratio

            not_self = torch.ones_like(distance)
            not_self[torch.arange(0, len(distance)), torch.arange(0, len(distance))] = 0

            dis_loss = utils.mse_loss(distance[not_self == 1], gt_distance[not_self == 1], self.hparams.weighted_method)
            loss = self.dis_loss_w * dis_loss * self.hparams.dis_loss_ratio

            if loss.isnan().any():
                breakpoint()

            self.val_loss += loss

            return {'loss': loss}

    def training_epoch_end(
            self,
            outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        self.dis_loss_w = 100 + 1e-3 * (self.trainer.current_epoch - 1e4) * (self.trainer.current_epoch > 1e4)
        self.counting += 1
        if self.trainer.current_epoch % 100 == 0:
            if self.training_classifier:
                subdir = 'classifier'
            else:
                subdir = self.current_model
            os.makedirs(f'{self.hparams.model_dir}/{subdir}', exist_ok=True)
            self.trainer.save_checkpoint(f'{self.hparams.model_dir}/{subdir}/epoch-{self.trainer.current_epoch}.pth')
            if self.trainer.current_epoch > 0:
                try:
                    os.remove(f'{self.hparams.model_dir}/{subdir}/epoch-{self.trainer.current_epoch - 100}.pth')
                except:
                    pass

    def configure_optimizers(self):
        if self.current_model == -1:
            params = list(self.classifier.conv.parameters()) + list(self.classifier.resblocks.parameters()) + list(self.classifier.linear.parameters())
        else:
            params = list(self.encoder.conv.parameters()) + list(self.encoder.resblocks.parameters()) + list(self.encoder.linear.parameters())
        return torch.optim.Adam(params,
                                lr=self.hparams.lr)

    def train_dataloader(self) -> DataLoader:
        # self.train_data = self.train_data_list[0]
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
        # self.train_data = data.data(self.hparams, calculate_distance_matrix=True)
        # self.train_data_list = data.make_datalist(self.hparams)
        if not self.training_classifier:
            print(f'training {self.current_model} model...')
            self.train_data = data.get_data(self.current_model, self.hparams)
        else:
            # self.train_data = self.train_data_list[-1]
            print(f'training classifier...')
            self.train_data = data.get_data(-1, self.hparams)
        loader = DataLoader(self.train_data,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=self.hparams.num_worker,
                            drop_last=True)
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

        if self.counting == self.hparams.lr_update_freq:
            # if (epoch + 1) % self.hparams.lr_decay == 0:
            # lr = 3e-5 + self.hparams.lr * (0.1 ** ((self.trainer.current_epoch + 1) / self.hparams.lr_decay))
            # print('lr', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1 + 2e-5
            print('lr', param_group['lr'])
            print(self.counting, self.hparams.lr_update_freq)
            self.counting = 0


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
