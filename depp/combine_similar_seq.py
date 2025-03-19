#!/usr/bin/env python3
import json
import os

import torch
import argparse
from depp import utils
import treeswift as ts
import numpy as np
import pandas as pd
from Bio import SeqIO

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--tree', type=str)
parser.add_argument('--seq', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--cluster', type=str, default=None)
parser.add_argument('--thr', type=str, default=None)
parser.add_argument('--replicate-seq', action='store_true')
parser.add_argument('--gap-encode', default=1/4)
args = parser.parse_args()

def process_seq(self_seq, args, need_mask=False):
    L = len(list(self_seq.values())[0])
    names = list(self_seq.keys())
    seqs = np.zeros([4, len(self_seq), L])
    if need_mask:
        mask = np.ones([1, len(self_seq), L])
    raw_seqs = [np.array(self_seq[k].seq).reshape(1, -1) for k in self_seq]
    raw_seqs = np.concatenate(raw_seqs, axis=0)
    seqs[0][raw_seqs == 'A'] = 1
    seqs[1][raw_seqs == 'C'] = 1
    seqs[2][raw_seqs == 'G'] = 1
    seqs[3][raw_seqs == 'T'] = 1

    # R
    idx = raw_seqs == 'R'
    seqs[0][idx] = 1 / 2
    seqs[2][idx] = 1 / 2

    # Y
    idx = raw_seqs == 'Y'
    seqs[1][idx] = 1 / 2
    seqs[3][idx] = 1 / 2

    # S
    idx = raw_seqs == 'S'
    seqs[1][idx] = 1 / 2
    seqs[2][idx] = 1 / 2

    # W
    idx = raw_seqs == 'W'
    seqs[0][idx] = 1 / 2
    seqs[3][idx] = 1 / 2

    # K
    idx = raw_seqs == 'K'
    seqs[2][idx] = 1 / 2
    seqs[3][idx] = 1 / 2

    # M
    idx = raw_seqs == 'M'
    seqs[0][idx] = 1 / 2
    seqs[1][idx] = 1 / 2

    # B
    idx = raw_seqs == 'B'
    seqs[1][idx] = 1 / 3
    seqs[2][idx] = 1 / 3
    seqs[3][idx] = 1 / 3

    # D
    idx = raw_seqs == 'D'
    seqs[0][idx] = 1 / 3
    seqs[2][idx] = 1 / 3
    seqs[3][idx] = 1 / 3

    # H
    idx = raw_seqs == 'H'
    seqs[0][idx] = 1 / 3
    seqs[1][idx] = 1 / 3
    seqs[3][idx] = 1 / 3

    # V
    idx = raw_seqs == 'V'
    seqs[0][idx] = 1 / 3
    seqs[1][idx] = 1 / 3
    seqs[2][idx] = 1 / 3

    seqs[:, raw_seqs == '-'] = args.gap_encode
    seqs[:, raw_seqs == 'N'] = args.gap_encode

    if need_mask:
        mask[:, raw_seqs == '-'] = 0
        mask[:, raw_seqs == 'N'] = 0
        mask = np.transpose(mask, axes=(1, 0, 2))

    seqs = np.transpose(seqs, axes=(1, 0, 2))
    if args.replicate_seq:
        df = pd.DataFrame(columns=['seqs'])
        df['seqs'] = df['seqs'].astype(object)
        df['seqs'] = list(seqs)
        df['names'] = names
        df = df.set_index('names')
        df = df.groupby(by=lambda x: x.split('_')[0]).sum(numeric_only=False)
        seqs = np.concatenate([i.reshape(1, 4, -1) for i in df['seqs'].values])
        seqs /= (seqs.sum(1, keepdims=True) + 1e-8)
        comb_names = list(df.index)
        if need_mask:
            mask_df = pd.DataFrame(columns=['masks'])
            mask_df['masks'] = mask_df['masks'].astype(object)
            mask_df['masks'] = list(mask)
            mask_df['names'] = names
            mask_df = mask_df.set_index('names')
            mask_df = mask_df.groupby(by=lambda x: x.split('_')[0]).sum(numeric_only=False)
            mask_df = mask_df.loc[comb_names]
            mask = np.concatenate([i.reshape(1, 1, -1) for i in mask_df['masks'].values])
        names = comb_names

    if need_mask:
        return names, torch.from_numpy(seqs), torch.from_numpy(mask).bool()
    return names, torch.from_numpy(seqs)

if __name__ == "__main__":
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
    seq = SeqIO.to_dict(SeqIO.parse(args.seq, "fasta"))
    tree = ts.read_tree_newick(args.tree)
    names, seq_array = process_seq(seq, args=args)
    seq_array = dict(zip(names, seq_array.numpy()))

    modified_tree, LCA_node_name = utils.combine_identical_seq_intree(tree=tree, seq=seq_array, outdir=args.outdir,
                                                                      thr=args.thr, cluster=args.cluster,
                                                                      replicate=args.replicate_seq)

    modified_tree.write_tree_newick(f'{args.outdir}/tree_fortrain.newick')
    new_seq = {}
    copy_count = {}
    mapping = {}
    for i in seq:
        if args.replicate_seq:
            species_name = i.split('_')[0]
        else:
            species_name = i
        if LCA_node_name[species_name] == species_name:
            new_seq[i] = str(seq[i].seq)
            mapping[i] = i
        else:
            new_name = f'{LCA_node_name[species_name]}splithere{copy_count.get(LCA_node_name[species_name], 0)}'
            copy_count[LCA_node_name[species_name]] = copy_count.get(LCA_node_name[species_name], 0) + 1
            new_seq[new_name] = str(seq[i].seq)
            mapping[new_name] = i

    seq_str = "".join([f">{i}\n{new_seq[i]}\n" for i in new_seq])
    with open(f'{args.outdir}/backbone_seq_lca.fa', 'w') as f:
        f.write(seq_str)

    with open(f'{args.outdir}/mapping.json', 'w') as f:
        json.dump(mapping, f)