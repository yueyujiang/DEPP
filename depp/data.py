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

def make_datalist(args):
    datasets = []
    for i in range(args.cluster_num):
        backbone_tree_file = f"{args.treedir}/{i}.nwk"
        backbone_seq_file = f"{args.seqdir}/{i}.fa"
        datasets.append(data(args, backbone_tree_file, backbone_seq_file, idx=i, calculate_distance_matrix=True))
    datasets.append(data_agg(args))
    return datasets

def get_data(i, args):
    if i == -1:
        return data_agg(args)
    else:
        if args.backbone_tree_file is None and args.backbone_seq_file is None:
            backbone_tree_file = f"{args.treedir}/{i}.nwk"
            backbone_seq_file = f"{args.seqdir}/{i}.fa"
        else:
            backbone_tree_file = args.backbone_tree_file
            backbone_seq_file = args.backbone_seq_file
        return data(args, backbone_tree_file, backbone_seq_file, idx=i, calculate_distance_matrix=True)

class data(Dataset):
    def __init__(self, args, backbone_tree_file, backbone_seq_file, idx, calculate_distance_matrix=False):
        self.args = args
        print('Loding data...')
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
        self.model_idx = idx

    def true_distance(self, nodes1, nodes2):
        # gt_distance_list = itemgetter(*nodes1)(self.distance_matrix)
        # gt_distance = [torch.tensor(itemgetter(*nodes2)(item)).view(1, len(nodes2)) for item in gt_distance_list]
        # gt_distance = torch.cat(gt_distance, dim=0)
        gt_distance = self.distance_matrix.loc[nodes1][nodes2]
        return torch.from_numpy(gt_distance.values)

    def __getitem__(self, idx):
        if self.args.add_random_mask:
            sample = {}
            node_name = self.nodes[idx]
            seq = self.seq[node_name]
            # sample['seqs'] = seq
            sample['nodes'] = node_name
            p = np.random.rand() * 0.2
            method = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            size = int(p * seq.shape[-1])
            if method == 1:
                start = np.random.choice(seq.shape[-1])
                mask = torch.ones(1, seq.shape[-1])
                mask[:, start: start + size] = 0
            elif method == 2:
                mask_site = np.random.choice(seq.shape[-1], size=size)
                mask = torch.ones(1, seq.shape[-1])
                mask[:, mask_site] = 0
            elif method == 0:
                mask = torch.ones(1, seq.shape[-1])
        # sample['masked_seqs'] = mask * seq
            sample['seqs'] = seq
            sample['seqs'][(mask != 1).repeat(4, 1)] = self.args.gap_encode
            sample['cluster_idx'] = self.model_idx
            return sample
        else:
            sample = {}
            node_name = self.nodes[idx]
            seq = self.seq[node_name]
            sample['seqs'] = seq
            sample['nodes'] = node_name
            sample['cluster_idx'] = self.model_idx
            return sample

    def __len__(self):
        return len(self.nodes)

class data_agg(Dataset):
    def __init__(self, args, calculate_distance_matrix=False):
        self.args = args

        self.total_seq = 0
        self.all_seq = {}
        self.all_nodes = []
        self.all_idx = []
        for i in range(args.cluster_num):
            print(f'loading sequences {i}...')
            backbone_seq_file = f'{args.seqdir}/{i}.fa'
            self_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
            args.sequence_length = len(list(self_seq.values())[0])
            L = args.sequence_length
            nodes, seq, mask = utils.process_seq(self_seq, args, True, True)
            self.all_nodes += list(nodes)
            self.all_seq.update(dict(zip(nodes, seq)))
            self.all_idx += [i] * len(seq)
        self.current_class = None

    def __getitem__(self, idx):
        if self.args.add_random_mask:
            sample = {}
            node_name = self.all_nodes[idx]
            seq = self.all_seq[node_name]
            model_idx = self.all_idx[idx]
            # sample['seqs'] = seq
            sample['nodes'] = node_name
            p = np.random.rand() * 0.2
            method = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            size = int(p * seq.shape[-1])
            if method == 1:
                start = np.random.choice(seq.shape[-1])
                mask = torch.ones(1, seq.shape[-1])
                mask[:, start: start + size] = 0
            elif method == 2:
                mask_site = np.random.choice(seq.shape[-1], size=size)
                mask = torch.ones(1, seq.shape[-1])
                mask[:, mask_site] = 0
            elif method == 0:
                mask = torch.ones(1, seq.shape[-1])
        # sample['masked_seqs'] = mask * seq
            sample['seqs'] = seq
            sample['seqs'][(mask != 1).repeat(4, 1)] = self.args.gap_encode
            sample['cluster_idx'] = model_idx
            return sample
        else:
            sample = {}
            node_name = self.all_nodes[idx]
            seq = self.all_seq[node_name]
            model_idx = self.all_idx[idx]
            sample['seqs'] = seq
            sample['nodes'] = node_name
            sample['cluster_idx'] = model_idx
            return sample

    def __len__(self):
        return len(self.all_nodes)


# class data(Dataset):
#     def __init__(self, args, calculate_distance_matrix=False):
#         self.args = args
#         print('Loding data...')
#
#         #        self.nodes = list(self_seq.keys())
#
#         print('finish data loading!')
#
#         if calculate_distance_matrix:
#             self.distance_matrix = []
#             print('Calculating distance matrix...')
#             for i in range(args.cluster_num):
#                 print(f'tree {i}')
#                 tree = treeswift.read_tree_newick(f'{args.treedir}/{i}.nwk')
#                 distance_matrix = tree.distance_matrix(leaf_labels=True)
#                 for key in distance_matrix:
#                     distance_matrix[key][key] = 0
#                 self.distance_matrix.append(pd.DataFrame.from_dict(distance_matrix))
#             print('Finish distance matrix calculation!')
#
#         self.seq = {}
#         self.nongaps = {}
#         self.num = {}
#         self.nodes = {}
#         self.total_seq = 0
#         self.all_seq = {}
#         self.all_nodes = []
#         self.all_idx = []
#         for i in range(args.cluster_num):
#             print(f'loading sequences {i}...')
#             backbone_seq_file = f'{args.seqdir}/{i}.fa'
#             self_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
#             args.sequence_length = len(list(self_seq.values())[0])
#             L = args.sequence_length
#             nodes, seq, mask = utils.process_seq(self_seq, args, True, True)
#             self.seq[i] = dict(zip(nodes, seq))
#             self.nongaps[i] = dict(zip(nodes, mask))
#             self.num[i] = len(nodes)
#             self.nodes[i] = nodes
#             self.total_seq += len(seq)
#             self.all_nodes += list(nodes)
#             self.all_seq.update(self.seq[i])
#             self.all_idx += [i] * len(seq)
#         self.current_class = None
#
#     def true_distance(self, nodes1, nodes2, model_idx):
#         # gt_distance_list = itemgetter(*nodes1)(self.distance_matrix)
#         # gt_distance = [torch.tensor(itemgetter(*nodes2)(item)).view(1, len(nodes2)) for item in gt_distance_list]
#         # gt_distance = torch.cat(gt_distance, dim=0)
#         gt_distance = self.distance_matrix[model_idx].loc[nodes1][nodes2]
#         return torch.from_numpy(gt_distance.values)
#
#     def __getitem__(self, idx):
#         if self.current_class is None:
#             sample = {}
#             node_name = self.all_nodes[idx]
#             seq = self.all_seq[node_name]
#             model_idx = self.all_idx[idx]
#         else:
#             idx = idx % len(self.seq[self.current_class])
#             model_idx = self.current_class
#             sample = {}
#             node_name = self.nodes[model_idx][idx]
#             seq = self.seq[model_idx][node_name]
#         sample['seqs'] = seq
#         sample['nodes'] = node_name
#         sample['cluster_idx'] = model_idx
#         return sample
#
#     def __len__(self):
#         return self.total_seq

# class data_mask(Dataset):
#     def __init__(self, args, calculate_distance_matrix=False):
#         self.args = args
#         print('Loding data...')
#
# #        self.nodes = list(self_seq.keys())
#
#         print('finish data loading!')
#
#         if calculate_distance_matrix:
#             self.distance_matrix = []
#             print('Calculating distance matrix...')
#             for i in range(self.cluster_num):
#                 tree = treeswift.read_tree_newick(f'{args.treedir}/{i}.nwk')
#                 distance_matrix = tree.distance_matrix(leaf_labels=True)
#                 for key in self.distance_matrix:
#                     self.distance_matrix[key][key] = 0
#                 self.distance_matrix.append(pd.DataFrame.from_dict(self.distance_matrix))
#             print('Finish distance matrix calculation!')
#
#         self.seq = {}
#         self.nongaps = {}
#         self.num = {}
#         for i in range(self.cluster_num):
#             backbone_seq_file = f'{args.seqdir}/{i}.fa'
#             self_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
#             args.sequence_length = len(list(self_seq.values())[0])
#             L = args.sequence_length
#             nodes, seq, mask = utils.process_seq(self_seq, args, True, True)
#             self.seq[i] = dict(zip(nodes, seq))
#             self.nongaps[i] = dict(zip(nodes, mask))
#             self.num[i] = len(nodes)
#
#     def true_distance(self, nodes1, nodes2):
#         # gt_distance_list = itemgetter(*nodes1)(self.distance_matrix)
#         # gt_distance = [torch.tensor(itemgetter(*nodes2)(item)).view(1, len(nodes2)) for item in gt_distance_list]
#         # gt_distance = torch.cat(gt_distance, dim=0)
#         gt_distance = self.distance_matrix.loc[nodes1][nodes2]
#         return torch.from_numpy(gt_distance.values)
#
#     def __getitem__(self, idx):
#         sample = {}
#         node_name = self.nodes[idx]
#         seq = self.seq[node_name]
#         # sample['seqs'] = seq
#         sample['nodes'] = node_name
#         nongap = torch.arange(seq.shape[-1])
#         nongap = nongap[self.nongaps[node_name][0]]
#         if len(nongap)/seq.shape[-1] > 0.3:
#             p = np.random.rand() * (len(nongap)/seq.shape[-1] * 0.4)
#             method = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
#             size = int(p * seq.shape[-1])
#             if method == 1:
#                 start = np.random.choice(nongap)
#                 mask = torch.ones(1, seq.shape[-1])
#                 mask[:, start: start + size] = 0
#             elif method == 2:
#                 mask_site = np.random.choice(nongap, size=size)
#                 mask = torch.ones(1, seq.shape[-1])
#                 mask[:, mask_site] = 0
#             elif method == 0:
#                 mask = torch.ones(1, seq.shape[-1])
#         else:
#             mask = torch.ones(1, seq.shape[-1])
#         # sample['masked_seqs'] = mask * seq
#         sample['seqs'] = seq
#         sample['seqs'][(mask != 1).repeat(4, 1)] = self.args.gap_encode
#         return sample
#
#     def __len__(self):
#         return self.num

