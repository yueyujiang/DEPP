#!/usr/bin/env python3
import json

import torch
import torch.nn as nn
from depp import utils
from depp import Model_recon
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Callable, Union
import numpy as np
import math
import omegaconf
import sys


class model(LightningModule):
    def __init__(self, args, load_model=False, classifier_cluster_num=None):
        super(model, self).__init__()
        self.save_hyperparameters(args)
        if not self.hparams.sequence_length:
            utils.get_seq_length(self.hparams)
        embedding_size = self.hparams.embedding_size
        self.encoders = []
        for i in range(self.hparams.cluster_num):
            if isinstance(self.hparams.embedding_size, omegaconf.listconfig.ListConfig) or \
                    isinstance(self.hparams.embedding_size, list):
                self.hparams.embedding_size = embedding_size[i]
                self.encoders.append(Model_recon.encoder(self.hparams))
                self.hparams.embedding_size = []
            else:
                self.encoders.append(Model_recon.encoder(self.hparams))
        self.classifier = Model_recon.classifier(self.hparams, cluster_num=classifier_cluster_num)

        if load_model:
            for i in range(self.hparams.cluster_num):
                print(f'loading {i} model...')
                save_epoch_num = self.hparams.epoch - self.hparams.epoch % 100
                sub_model = Model_recon.model.load_from_checkpoint(
                    f'{self.hparams.model_dir}/{i}/epoch-{save_epoch_num}.pth', current_model=i)
                # breakpoint()
                self.encoders[i].load_state_dict(sub_model.encoder.state_dict(), strict=True)
            classifier = Model_recon.model.load_from_checkpoint(
                f'{self.hparams.model_dir}/classifier/epoch-{self.hparams.classifier_epoch - self.hparams.classifier_epoch % 100}.pth', current_model=-1)
            self.classifier.load_state_dict(classifier.classifier.state_dict(), strict=True)

        self.hparams.embedding_size = embedding_size
        self.encoders = nn.ModuleList(self.encoders)
        self.channel = 4
        self.hparams.distance_ratio = math.sqrt(float(1.0 / 128 / 10 * float(self.hparams.distance_alpha)))

        self.dis_loss_w = 100
        self.train_loss = []
        self.val_loss = float('inf')

        self.testing = False
        if self.hparams.classifier_epoch == 0:
            self.training_classifier = False
            self.current_model = -1
        else:
            self.training_classifier = True
            self.current_model = args.start_model_idx
        self.counting = 0

        if load_model:
            with open(args.cluster_corr, 'r') as f:
                self.hparams.corr_str = f.read()
        else:
            corr = json.loads(self.hparams.corr_str)
            self.corr = {int(i): torch.tensor(corr[i]) for i in corr}

        self.save_hyperparameters(self.hparams)

    def forward(self, x, model_idx=None, only_class=False) -> torch.Tensor:
        if self.testing:
            model_prob = self.classifier(x).softmax(-1)
            model_prob = torch.stack([model_prob[:, self.corr[i]].max(-1)[0] for i in range(self.hparams.cluster_num)], dim=-1)
            # model_prob = model_prob / model_prob.sum(-1, keepdims=True)
            model_idx = model_prob.argmax(-1)
            if only_class:
                return model_idx, model_prob
            result_tmp = [self.encoders[model_idx[i]](x[i].unsqueeze(0)) for i in range(len(x))]
            result = np.empty(len(result_tmp), dtype=object)
            result[:] = result_tmp
            return result, model_idx, model_prob
        if self.training_classifier:
            return self.classifier(x)
        return self.encoders[model_idx](x)

    def configure_optimizers(self):
        self.trainer.save_checkpoint(
            f'{self.hparams.model_dir}/cluster-depp.pth')
        sys.exit()

    def on_epoch_start(self) -> None:
        # self.trainer.save_checkpoint(
        #     f'{self.hparams.model_dir}/epoch-agg.pth')
        sys.exit()

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(
            self,
            outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        pass

    def configure_optimizers(self):
        # self.trainer.save_checkpoint(
        #     f'{self.hparams.model_dir}/epoch-agg.pth')
        return None

    def train_dataloader(self) -> DataLoader:
        pass

    def validation_step(self, batch, batch_idx):
        return {}

    def val_dataloader(self):
        # TODO: do a real train/val split
        self.trainer.save_checkpoint(
            f'{self.hparams.model_dir}/cluster-depp.pth')
        sys.exit()
