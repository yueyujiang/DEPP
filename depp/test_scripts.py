import numpy as np
# import utils
import torch
import treeswift as ts
from depp import utils
from Bio import SeqIO

seq = SeqIO.to_dict(SeqIO.parse('/Users/yueyu/Documents/bioinformatics/project/gut_data_tmp/kmer_based_placement_data2_noskin/seq_500.fa', "fasta"))
tree = ts.read_tree_newick('/Users/yueyu/Documents/bioinformatics/project/gut_data_tmp/kmer_based_placement_data2_noskin/wol.nwk')

def process_seq(self_seq):
    L = len(list(self_seq.values())[0])
    seq_tmp = {}
    raw_seqs = []
    ks = []
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
        seq[:, raw_seq == '-'] = 1/4
        seq_tmp[k.split('_')[0]] += seq
    for k in seq_tmp:
        seq_tmp[k] = seq_tmp[k].float() / (seq_tmp[k].sum(dim=0, keepdim=True) + 1e-8)
    names = []
    seqs = []
    for k in seq_tmp:
        names.append(k)
        seqs.append(seq_tmp[k].unsqueeze(0))
    return names, torch.cat(seqs, dim=0)

names, seq = process_seq(seq)
seq = dict(zip(names, seq.numpy()))

modified_tree = utils.combine_identical_seq_intree(tree=tree, seq=seq,
                                                   outdir='~/Documents/bioinformatics/project/gut_data_tmp/kmer_based_placement_data2_noskin',
                                                   thr=0.1)