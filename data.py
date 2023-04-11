import os
import torch
import treeswift
import numpy as np
import pandas as pd
import utils
from torch.utils.data import Dataset
from operator import itemgetter
from Bio import SeqIO

class data(Dataset):
    def __init__(self, args, calculate_distance_matrix=False):
        if not os.path.exists('large_tree_tmp'):
            os.makedirs('large_tree_tmp')
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
            if args.internal:
                self.distance_matrix = pd.read_csv(args.distance_file, sep='\t', index_col=0)
            else:
                self.distance_matrix = tree.distance_matrix(leaf_labels=True)
                for key in self.distance_matrix:
                    self.distance_matrix[key][key] = 0
                self.distance_matrix = pd.DataFrame.from_dict(self.distance_matrix)
            print('Finish distance matrix calculation!')

        seq_tmp = {}
        raw_seqs = []
        ks = []
        if args.replicate_seq:
            for k in self_seq:
                seq_tmp[k.split('_')[0]] = torch.zeros(4, L)
        for k in self_seq:
            seq = np.zeros([4, L])
            raw_seq = np.array(self_seq[k])
            raw_seqs.append(raw_seq.reshape(1, -1))
            ks.append(k)
            seq[0][raw_seq == 'A'] = 1
            seq[1][raw_seq == 'C'] = 1
            seq[2][raw_seq == 'G'] = 1
            seq[3][raw_seq == 'T'] = 1
            seq[:, raw_seq == '-'] = args.gap_encode
            if args.replicate_seq:
                seq_tmp[k.split('_')[0]] += torch.from_numpy(seq)
            else:
                seq_tmp[k] = torch.from_numpy(seq)
        if args.replicate_seq:
            for k in seq_tmp:
                seq_tmp[k] = seq_tmp[k].float() / (seq_tmp[k].sum(dim=0, keepdim=True) + 1e-8)
        self.seq = seq_tmp
        self.nodes = list(self.seq.keys())
        print('jc_correct', args.jc_correct)
        if args.jc_correct:
            # breakpoint()n
            raw_seqs = np.concatenate(raw_seqs, axis=0)
            jc_dist = utils.jc_dist(raw_seqs, raw_seqs, ks, ks)
            jc_dist = jc_dist.groupby(by=lambda x: x.split('_')[0], axis=0).min()
            jc_dist = jc_dist.groupby(by=lambda x: x.split('_')[0], axis=1).min()
            # breakpoint()
            ks = self.distance_matrix.index
            jc_dist = jc_dist.loc[ks][ks]
            self.distance_matrix = self.distance_matrix[ks]
            # breakpoint()
            args.jc_ratio = float((self.distance_matrix[jc_dist!=0] / jc_dist[jc_dist!=0]).min().min()) / 2
            print(self.distance_matrix['G000005825']['G000006175'], jc_dist['G000005825']['G000006175'], "args.ratio", args.jc_ratio)
            self.distance_matrix -= args.jc_ratio * jc_dist
            print(self.distance_matrix['G000005825']['G000006175'])

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
