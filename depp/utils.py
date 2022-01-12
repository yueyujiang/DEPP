#!/usr/bin/env python3

import torch
import os
import pandas as pd
import math
import numpy as np
import dendropy
import csv
import time
from Bio import SeqIO


def get_seq_length(args):
    backbone_seq_file = args.backbone_seq_file
    backbone_tree_file = args.backbone_tree_file
    seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
    args.sequence_length = len(list(seq.values())[0])
    tree = dendropy.Tree.get(path=backbone_tree_file, schema='newick')
    num_nodes = len(tree.leaf_nodes())
    if args.embedding_size == -1:
        args.embedding_size = 2 ** math.floor(math.log2(10 * num_nodes ** (1 / 2)))


def distance_portion(nodes1, nodes2, mode):
    if len(nodes1.shape) == 1:
        nodes1 = nodes1.unsqueeze(0)
    if len(nodes2.shape) == 1:
        nodes2 = nodes2.unsqueeze(0)
    n1 = len(nodes1)
    n2 = len(nodes2)
    nodes1 = nodes1.view(n1, 1, -1)
    nodes2 = nodes2.view(1, n2, -1)
    if mode == 'ms':
        return torch.sum((nodes1 - nodes2) ** 2, dim=-1)
    elif mode == 'L2':
        # breakpoint()
        return (torch.sum((nodes1 - nodes2) ** 2, dim=-1) + 1e-6).sqrt()
    elif mode == 'L1':
        return torch.sum(abs(nodes1 - nodes2), dim=-1)
    elif mode == 'cosine':
        return 1 - torch.nn.functional.cosine_similarity(nodes1, nodes2, dim=-1)
    elif mode == 'tan':
        cosine = torch.nn.functional.cosine_similarity(nodes1, nodes2, dim=-1)
        return (1 - cosine ** 2) / (cosine + 1e-9)


def distance(nodes1, nodes2, mode):
    # node1: query
    # node2: backbone
    dist = torch.cat([torch.cat(
        [distance_portion(nodes1[j * 1000: (j + 1) * 1000], nodes2[i * 1000: (i + 1) * 1000], mode) for j in
         range(math.ceil(len(nodes1) / 1000))],
        dim=0) for i in range(math.ceil(len(nodes2) / 1000))], dim=1)
    return dist


def mse_loss(model_dist, true_dist, weighted_method):
    assert model_dist.shape == true_dist.shape
    if weighted_method == 'ols':
        return ((model_dist - true_dist) ** 2).mean()
    elif weighted_method == 'fm':
        weight = 1 / (true_dist + 1e-4) ** 2
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'be':
        weight = 1 / (true_dist + 1e-4)
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_fm':
        true_dist = torch.sqrt(true_dist)
        weight = 1 / (true_dist + 1e-4) ** 2
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_be':
        true_dist = torch.sqrt(true_dist)
        weight = 1 / (true_dist + 1e-4)
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_ols':
        true_dist = torch.sqrt(true_dist)
        weight = 1
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_sqrt':
        true_dist = torch.sqrt(true_dist)
        weight = 1 / (torch.sqrt(true_dist) + 1e-4)
        return ((model_dist - true_dist) ** 2 * weight).mean()
    elif weighted_method == 'square_root_four':
        true_dist = torch.sqrt(true_dist)
        weight = 1 / (true_dist + 1e-4) ** 4
        return ((model_dist - true_dist) ** 2 * weight).mean()


def process_seq(self_seq, args, isbackbone, need_mask=False):
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
    if args.replicate_seq and (isbackbone or args.query_dist):
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

def get_embeddings(seqs, model, mask=None):
    encodings = []
    for i in range(math.ceil(len(seqs) / 2000.0)):
        if not (mask is None):
            encodings_tmp = model(seqs[i * 2000: (i + 1) * 2000].float(), mask=mask[i * 2000: (i + 1) * 2000]).detach()
        else:
            encodings_tmp = model(seqs[i * 2000: (i + 1) * 2000].float()).detach()
        encodings.append(encodings_tmp)
    encodings = torch.cat(encodings, dim=0)
    return encodings

def save_depp_dist(model, args, recon_model=None):
    t1 = time.time()
    model.eval()
    print('processing data...')
    backbone_seq_file = args.backbone_seq_file
    query_seq_file = args.query_seq_file
    dis_file_root = os.path.join(args.outdir)
    # args.distance_ratio = float(1.0 / float(args.embedding_size) / 10 * float(args.distance_alpha))
    args.distance_ratio = model.hparams.distance_ratio
    args.gap_encode = model.hparams.gap_encode
    args.jc_correct = model.hparams.jc_correct
    args.replicate_seq = model.hparams.replicate_seq
    print('jc_correct', args.jc_correct)
    if args.jc_correct:
        args.jc_ratio = model.hparams.jc_ratio
    if not os.path.exists(dis_file_root):
        os.makedirs(dis_file_root)

    backbone_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
    query_seq = SeqIO.to_dict(SeqIO.parse(query_seq_file, "fasta"))

    if args.jc_correct:
        backbone_seq_names, backbone_seq_names_raw, backbone_seq_tensor, backbone_raw_array = \
            process_seq(backbone_seq, args, isbackbone=True)
        query_seq_names, query_seq_names_raw, query_seq_tensor, query_raw_array = \
            process_seq(query_seq, args, isbackbone=False)
    else:
        # breakpoint()
        if not (recon_model is None):
            backbone_seq_names, backbone_seq_tensor, backbone_mask = process_seq(backbone_seq, args, isbackbone=True, need_mask=True)
            query_seq_names, query_seq_tensor, query_mask = process_seq(query_seq, args, isbackbone=False, need_mask=True)
        else:
            backbone_seq_names, backbone_seq_tensor = process_seq(backbone_seq, args, isbackbone=True)
            query_seq_names, query_seq_tensor = process_seq(query_seq, args, isbackbone=False)

    for param in model.parameters():
        param.requires_grad = False
    print('finish data processing!')
    print(f'{len(backbone_seq_names)} backbone sequences')
    print(f'{len(query_seq_names)} query sequence(s)')
    print(f'calculating embeddings...')
    backbone_encodings = get_embeddings(backbone_seq_tensor, model)
    query_encodings = get_embeddings(query_seq_tensor, model)
    torch.save(query_encodings, f'{dis_file_root}/query_embeddings.pt')
    torch.save(query_seq_names, f'{dis_file_root}/query_names.pt')

    if not (recon_model is None):
        recon_backbone_encodings = get_embeddings(backbone_seq_tensor, recon_model, backbone_mask)
        recon_query_encodings = get_embeddings(query_seq_tensor, recon_model, query_mask)

    print(f'finish embedding calculation!')
    print(f'calculating distance matrix...')
    t2 = time.time()
    #print('calculate embeddings', t2 - t1)

    # query_dist = distance(query_encodings, backbone_encodings, args.distance_mode) * args.distance_ratio
    query_dist = distance(query_encodings, backbone_encodings, args.distance_mode) * args.distance_ratio
    if 'square_root' in args.weighted_method:
        query_dist = query_dist ** 2

    if recon_model:
        gap_portion = 1 - query_mask.int().sum(-1) / query_mask.shape[-1]
        recon_query_dist = distance(recon_query_encodings, recon_backbone_encodings, args.distance_mode) * args.distance_ratio
        if 'square_root' in args.weighted_method:
            recon_query_dist = recon_query_dist ** 2
        query_dist = query_dist * (1 - gap_portion) + recon_query_dist * gap_portion

    t3 = time.time()
    #print('calculate distance', t3 - t2)
    query_dist = np.array(query_dist)
    query_dist[query_dist < 1e-3] = 0
    data_origin = dict(zip(query_seq_names, list(query_dist.astype(str))))
    data_origin = "\t" + "\t".join(backbone_seq_names) + "\n" + \
                  "\n".join([str(k) + "\t"+ "\t".join(data_origin[k]) for k in data_origin]) + "\n"
    t4 = time.time()
    #print('convert string', t4 - t3)
    with open(os.path.join(dis_file_root, f'depp.csv'), 'w') as f:
        f.write(data_origin)
    t5 = time.time()
    #print('save string', t5 - t4)
    # data_origin = pd.DataFrame.from_dict(data_origin, orient='index', columns=backbone_seq_names)

    if args.query_dist:
        idx = data_origin.index
        data_origin = data_origin[idx]
    # data_origin.to_csv(os.path.join(dis_file_root, f'depp.csv'), sep='\t')
    # if not os.path.isdir(f'{args.outdir}/depp_tmp'):
    #     os.makedirs(f'{args.outdir}/depp_tmp')
    # with open(f'{args.outdir}/depp_tmp/seq_name.txt', 'w') as f:
    #     f.write("\n".join(query_seq_names) + '\n')
    print('original distanace matrix saved!')
    print("take {:.2f} seconds".format(t5-t1))
