import os
import torch
import treeswift
import numpy as np
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

        self.nodes = list(self_seq.keys())

        print('finish data loading!')

        args.sequence_length = len(list(self_seq.values())[0])
        L = args.sequence_length

        if calculate_distance_matrix:
            print('Calculating distance matrix...')
            self.distance_matrix = tree.distance_matrix(leaf_labels=True)
            for key in self.distance_matrix:
                self.distance_matrix[key][key] = 0
            print('Finish distance matrix calculation!')

        seq_tmp = {}
        for k in self_seq:
            seq = np.zeros([4, L])
            raw_seq = np.array(self_seq[k])
            seq[0][raw_seq == 'A'] = 1
            seq[1][raw_seq == 'C'] = 1
            seq[2][raw_seq == 'G'] = 1
            seq[3][raw_seq == 'T'] = 1
            seq_tmp[k] = torch.from_numpy(seq)
        self.seq = seq_tmp

    def true_distance(self, nodes1, nodes2):
        gt_distance_list = itemgetter(*nodes1)(self.distance_matrix)
        gt_distance = [torch.tensor(itemgetter(*nodes2)(item)).view(1, len(nodes2)) for item in gt_distance_list]
        gt_distance = torch.cat(gt_distance, dim=0)
        return gt_distance

    def __getitem__(self, idx):
        sample = {}
        node_name = self.nodes[idx]
        seq = self.seq[node_name]
        sample['seqs'] = seq
        sample['nodes'] = node_name
        return sample

    def __len__(self):
        return len(self.nodes)
