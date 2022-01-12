#!/usr/bin/env python3

import os
import torch
import treeswift
import numpy as np
import pandas as pd
from depp import utils
from torch.utils.data import Dataset
from operator import itemgetter
from Bio import SeqIO

class data(Dataset):
    def __init__(self, args, calculate_distance_matrix=False):
        self.args = args
        print('Loding data...')
        backbone_tree_file = args.backbone_tree_file
        backbone_seq_file = args.backbone_seq_file
        self_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
        tree = treeswift.read_tree(backbone_tree_file, 'newick')

#        self.nodes = list(self_seq.keys())

        print('finish data loading!')

        args.sequence_length = len(list(self_seq.values())[0])
        L = args.sequence_length

        if calculate_distance_matrix:
            print('Calculating distance matrix...')
            self.distance_matrix = tree.distance_matrix(leaf_labels=True)
            for key in self.distance_matrix:
                self.distance_matrix[key][key] = 0
            self.distance_matrix = pd.DataFrame.from_dict(self.distance_matrix)
            print('Finish distance matrix calculation!')
        self.nodes, self.seq = utils.process_seq(self_seq, args, True)
        self.seq = dict(zip(self.nodes, self.seq))

    def true_distance(self, nodes1, nodes2):
        # gt_distance_list = itemgetter(*nodes1)(self.distance_matrix)
        # gt_distance = [torch.tensor(itemgetter(*nodes2)(item)).view(1, len(nodes2)) for item in gt_distance_list]
        # gt_distance = torch.cat(gt_distance, dim=0)
        gt_distance = self.distance_matrix.loc[nodes1][nodes2]
        return torch.from_numpy(gt_distance.values)

    def __getitem__(self, idx):
        sample = {}
        node_name = self.nodes[idx]
        seq = self.seq[node_name]
        sample['seqs'] = seq
        sample['nodes'] = node_name
        return sample

    def __len__(self):
        return len(self.nodes)
