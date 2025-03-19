#!/usr/bin/env python3
import collections

import torch
import os
import pandas as pd
import math
import numpy as np
import dendropy
import csv
import time
import json
import scipy.stats
import treeswift as ts
from Bio import SeqIO
from scipy.cluster.hierarchy import linkage, fcluster


def get_seq_length(args):
    if args.backbone_tree_file is not None and args.cluster_num == 1:
        backbone_seq_file = args.backbone_seq_file
        backbone_tree_file = args.backbone_tree_file
    else:
        backbone_seq_file = f'{args.seqdir}/0.fa'
        backbone_tree_file = f'{args.treedir}/0.nwk'
    seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
    args.sequence_length = len(list(seq.values())[0])
    tree = dendropy.Tree.get(path=backbone_tree_file, schema='newick')
    if args.embedding_size is None:
        args.embedding_size = []
        for i in range(args.cluster_num):
            if args.backbone_tree_file is not None and args.cluster_num == 1:
                file = backbone_tree_file
            else:
                file = f'{args.treedir}/{i}.nwk'
            tree = dendropy.Tree.get(path=file, schema='newick')
            num_nodes = len(tree.leaf_nodes())
            args.embedding_size.append(2 ** math.floor(math.log2(10 * num_nodes ** (1 / 2))))


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
    elif mode == 'hyperbolic':
        nodes1 = nodes1.squeeze(1)
        nodes2 = nodes2.squeeze(0)
        return hyp_dist(nodes1, nodes2)


def project_hyperbolic(x):
    N, d = x.shape[0], x.shape[1]

    hnorm_x = torch.norm(x, dim=1, p=2, keepdim=True)

    s1 = torch.cosh(hnorm_x)
    s2 = torch.div(torch.sinh(hnorm_x), hnorm_x)

    e = torch.zeros(N, d + 1).to(x.device)
    e[:, 0] = 1

    zero_col = torch.zeros(N, 1).to(x.device)
    z = torch.cat((zero_col, x), 1)
    z = torch.mul(s1, e) + torch.mul(s2, z)  # hyperbolic embeddings
    return z


def hyp_dist(embeddings1, embeddings2=None):
    if embeddings2 is None:
        x1 = x2 = project_hyperbolic(embeddings1)
    else:
        x1, x2 = project_hyperbolic(embeddings1), project_hyperbolic(embeddings2)
    d = x1.shape[1] - 1
    H = torch.eye(d + 1, d + 1).to(x1.device)
    H[0, 0] = -1
    N1, N2 = x1.shape[0], x2.shape[0]
    G = torch.matmul(torch.matmul(x1, H), torch.transpose(x2, 0, 1))
    G[G >= -1] = -1
    return torch.acosh(-G)


def distance(nodes1, nodes2, mode):
    # node1: query
    # node2: backbone
    dist = torch.cat([torch.cat(
        [distance_portion(nodes1[j * 100: (j + 1) * 100], nodes2[i * 100: (i + 1) * 100], mode) for j in
         range(math.ceil(len(nodes1) / 100))],
        dim=0) for i in range(math.ceil(len(nodes2) / 100))], dim=1)
    return dist


def mse_loss(model_dist, true_dist, weighted_method):
    assert model_dist.shape == true_dist.shape
    if weighted_method == 'ols':
        return ((model_dist - true_dist) ** 2).mean()
    elif weighted_method == 'fm':
        weight = 1 / (true_dist + 1e-8) ** 2
        return ((model_dist - true_dist) ** 2 * weight).mean().sqrt()
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


def get_embeddings_cluster(seqs, model, query=True, model_idx=None, only_class=False):
    with torch.no_grad():
        if query:
            model.testing = True
            if only_class:
                idxs = []
                idxs_probs = []
                for i in range(math.ceil(len(seqs) / 2000.0)):
                    idxs_tmp, idxs_prob_tmp = model(seqs[i * 2000: (i + 1) * 2000].float(), only_class=True)
                    idxs.append(idxs_tmp)
                    idxs_probs.append(idxs_prob_tmp)
                idxs = torch.cat(idxs)
                idxs_probs = torch.cat(idxs_probs, dim=0)
                return idxs, idxs_probs
            encodings = []
            idxs = []
            idxs_probs = []
            for i in range(math.ceil(len(seqs) / 2000.0)):
                encodings_tmp, idxs_tmp, idxs_prob_tmp = model(seqs[i * 2000: (i + 1) * 2000].float())
                encodings.append(encodings_tmp)
                idxs.append(idxs_tmp)
                idxs_probs.append(idxs_prob_tmp)
            encodings = np.concatenate(encodings)
            idxs = torch.cat(idxs)
            idxs_probs = torch.cat(idxs_probs, dim=0)
            return encodings, idxs, idxs_probs
        else:
            model.testing = False
            model.training_classifier = False
            encodings = []
            for i in range(math.ceil(len(seqs) / 2000.0)):
                encodings_tmp = model(seqs[i * 2000: (i + 1) * 2000].float(), model_idx).detach()
                encodings.append(encodings_tmp)
            encodings = torch.cat(encodings, dim=0)
            return encodings

def save_dataframe(data_origin, outfile):
    data_origin = data_origin.astype(str)
    data_origin = "\t" + "\t".join(data_origin.keys().astype(str)) + "\n" + \
                  "\n".join([str(k) + "\t" + "\t".join(data_origin.loc[k].values) for k in
                             data_origin.index]) + "\n"
    with open(outfile, 'w') as f:
        f.write(data_origin)

# @profile
def save_depp_dist_cluster(model, args, use_cluster=None, recon_model=None):
    t1 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    query_seq_file = args.query_seq_file
    args.replicate_seq = model.hparams.replicate_seq
    query_seq = SeqIO.to_dict(SeqIO.parse(query_seq_file, "fasta"))
    query_seq_names, query_seq_tensor = process_seq(query_seq, args, isbackbone=False, need_mask=False)
    print('calculating query embeddings...')
    if args.only_class:
        if recon_model is None:
            query_idxs, query_idxs_probs = get_embeddings_cluster(query_seq_tensor, model, query=True, only_class=True)
        else:
            query_idxs, query_idxs_probs = get_embeddings_cluster(query_seq_tensor, recon_model, query=True, only_class=True)
        with open(f'{args.outdir}/class.json', 'w') as f:
            tmp_class = dict(zip(query_seq_names, list(query_idxs.numpy().astype(int))))
            tmp_class = {i: int(tmp_class[i]) for i in tmp_class}
            json.dump(tmp_class, f, sort_keys=True, indent=4)
        torch.save(query_idxs_probs, f'{args.outdir}/class_prob.pt')
        torch.save(query_seq_names, f'{args.outdir}/query_seq_names.pt')
        t2 = time.time()
        print('finish! take {:.2f} seconds.'.format(t2 - t1))
        return
        # torch.save(query_idxs_probs, f'{args.outdir}/class_probs.pt')
        # torch.save(query_seq_names, f'{args.outdir}/query_labels.pt')

    if use_cluster is None:

        if args.use_multi_class:
            query_idxs, query_idxs_probs = get_embeddings_cluster(query_seq_tensor, model, query=True, only_class=True)
            sorted_probs, sorted_probs_idx = torch.sort(query_idxs_probs, dim=-1, descending=True)
            add_class_idx = (sorted_probs[:, 0] / sorted_probs[:, 1]) < args.prob_thr
            add_class_idx_third = ((sorted_probs[:, 1] / sorted_probs[:, 2]) < args.prob_thr) & add_class_idx
            add_class_idx_forth = ((sorted_probs[:, 2] / sorted_probs[:, 3]) < args.prob_thr) & add_class_idx_third

            cluster_idxs = {}
            for i in range(model.hparams.cluster_num):
                idx1 = sorted_probs_idx[:, 0] == i
                idx2 = (sorted_probs_idx[:, 1] == i) & add_class_idx
                idx3 = (sorted_probs_idx[:, 2] == i) & add_class_idx_third
                idx4 = (sorted_probs_idx[:, 3] == i) & add_class_idx_forth
                idx_all = idx1 | idx2 | idx3 | idx4
                if idx_all.sum() > 0:
                    cluster_idxs[i] = torch.arange(0, len(query_seq_tensor))[idx1 | idx2 | idx3 | idx4]

            query_encodings_dict = {i: get_embeddings_cluster(
                query_seq_tensor[cluster_idxs[i]],
                model,
                query=False,
                model_idx=i
                ) for i in cluster_idxs}
            query_names_dict = {i: list(np.array(query_seq_names)[cluster_idxs[i]]) if len(cluster_idxs[i]) > 1
                                            else [np.array(query_seq_names)[cluster_idxs[i]]] for i in cluster_idxs}
        else:
            query_encodings, query_idxs, query_idxs_probs = get_embeddings_cluster(query_seq_tensor, model, query=True)
            if len(query_encodings) == 1:
                query_encodings_dict = {query_idxs[0].item(): query_encodings[0]}
                query_names_dict = {query_idxs[0].item(): query_seq_names}
            else:
                query_encodings_dict = {i: torch.cat(list(query_encodings[query_idxs == i])) for i in
                                        range(model.hparams.cluster_num) if (query_idxs == i).sum() > 0}
                query_names_dict = {i: list(np.array(query_seq_names)[query_idxs == i]) for i in
                                    range(model.hparams.cluster_num) if (query_idxs == i).sum() > 0}
        # torch.save(query_idxs_probs, f'{args.outdir}/class_probs.pt')
        # torch.save(query_seq_names, f'{args.outdir}/query_labels.pt')
        entropy = scipy.stats.entropy(query_idxs_probs, axis=-1)
        entropy_s = [f'{query_seq_names[i]}\t{entropy[i]}\n' for i in range(len(entropy))]
        entropy_s = "".join(entropy_s)
        with open(f'{args.outdir}/entropy.txt', 'w') as f:
            f.write(entropy_s)
        torch.save(query_idxs_probs, f'{args.outdir}/prob.pt')

    else:
        query_encodings = get_embeddings_cluster(query_seq_tensor, model, query=False, model_idx=use_cluster)
        query_encodings_dict = {use_cluster: query_encodings}
        query_names_dict = {use_cluster: query_seq_names}
    t2 = time.time()
    print('finish query embeddings calculation! use {:.2f} seconds.'.format(t2 - t1))
    print('calculating backbone embeddings...')
    t3 = time.time()
    def get_backbone_embeddings(i):
        backbone_seq_file = f"{args.seqdir}/{i}.fa"
        backbone_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
        backbone_seq_names, backbone_seq_tensor = process_seq(backbone_seq, args, isbackbone=True)
        backbone_encodings = get_embeddings_cluster(backbone_seq_tensor, model, query=False, model_idx=i)
        return backbone_encodings, backbone_seq_names

    if args.backbone_emb is None:
        if use_cluster is None:
            backbone_embs_names_tmp = [get_backbone_embeddings(i) for i in range(model.hparams.cluster_num)]
        else:
            backbone_embs_names_tmp = [get_backbone_embeddings(use_cluster)]

        if use_cluster is None:
            backbone_encodings_dict = {i: backbone_embs_names_tmp[i][0] for i in range(model.hparams.cluster_num)}
            backbone_names_dict = {i: backbone_embs_names_tmp[i][1] for i in range(model.hparams.cluster_num)}
        else:
            backbone_encodings_dict = {use_cluster: backbone_embs_names_tmp[0][0]}
            backbone_names_dict = {use_cluster: backbone_embs_names_tmp[0][1]}
        torch.save(backbone_encodings_dict, f'{args.outdir}/backbone_emb.pt')
        torch.save(backbone_names_dict, f'{args.outdir}/backbone_ids.pt')
    else:
        backbone_names_dict = torch.load(args.backbone_id)
        backbone_encodings_dict = torch.load(args.backbone_emb)
    print('finish backbone embedding calculation! use {:.2f} seconds.'.format(t3 - t2))
    print('calculating distance matrix...')
    query_dist_dict = {}

    for i in query_encodings_dict:
        query_dist = distance(query_encodings_dict[i], backbone_encodings_dict[i],
                              args.distance_mode) * model.hparams.distance_ratio
        if 'square_root' in args.weighted_method:
            query_dist = query_dist ** 2
        query_dist = np.array(query_dist)
        query_dist[query_dist < 1e-3] = 0
        query_dist_dict[i] = query_dist
    t4 = time.time()
    print('finish distance calculation! use {:.2f} seconds.'.format(t4 - t3))

    if args.use_multi_class:
        query_dist_df_dict = \
            {i: pd.DataFrame.from_dict(dict(zip(query_names_dict[i], list(query_dist_dict[i])))) for i in query_dist_dict}
        for i in query_dist_df_dict:
            query_dist_df_dict[i].index = backbone_names_dict[i]

        for name_idx in range(math.ceil(len(query_seq_names) / 200)):
            cur_names = query_seq_names[name_idx*200: (name_idx+1)*200]
            data_origin = None
            for i in query_dist_dict:
                inter_names = list(set(cur_names).intersection(set(query_dist_df_dict[i].keys())))
                if len(inter_names) == 0:
                    continue
                if data_origin is None:
                    data_origin = query_dist_df_dict[i][inter_names]
                else:
                    data_origin = pd.concat([data_origin, query_dist_df_dict[i][inter_names]], axis=0)
            data_origin = data_origin.groupby(lambda x: x).median()
            data_origin = data_origin.fillna(-1)
            save_dataframe(data_origin.transpose(), f"{args.outdir}/depp{name_idx}.csv")
    else:
        for i in model.hparams.cluster_num:
            data_origin = dict(zip(query_names_dict[i], list(query_dist_dict[i].astype(str))))
            data_origin = "\t" + "\t".join(backbone_names_dict[i]) + "\n" + \
                          "\n".join([str(k) + "\t" + "\t".join(data_origin[k]) for k in data_origin]) + "\n"
            with open(f"{args.outdir}/depp{i}.csv", 'w') as f:
                f.write(data_origin)

        # print('convert string', t4 - t3)
        # with open(os.path.join(args.outdir, f'depp{i}.csv'), 'w') as f:
        #     f.write(data_origin)
    t2 = time.time()
    print('finish! take {:.2f} seconds.'.format(t2 - t1))


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
    args.replicate_seq = model.hparams.replicate_seq
    backbone_seq_file = args.backbone_seq_file
    query_seq_file = args.query_seq_file
    dis_file_root = os.path.join(args.outdir)
    # args.distance_ratio = float(1.0 / float(args.embedding_size) / 10 * float(args.distance_alpha))
    args.distance_ratio = model.hparams.distance_ratio
    args.gap_encode = model.hparams.gap_encode
    args.jc_correct = model.hparams.jc_correct
    # args.replicate_seq = model.hparams.replicate_seq
    print('jc_correct', args.jc_correct)
    if args.jc_correct:
        args.jc_ratio = model.hparams.jc_ratio
    if not os.path.exists(dis_file_root):
        os.makedirs(dis_file_root, exist_ok=True)

    if ((args.backbone_emb is None) or (args.backbone_id is None)) or (
            (recon_model is not None) and (args.recon_backbone_emb is None)):
        backbone_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
    else:
        backbone_seq = None
    query_seq = SeqIO.to_dict(SeqIO.parse(query_seq_file, "fasta"))

    if args.jc_correct:
        backbone_seq_names, backbone_seq_names_raw, backbone_seq_tensor, backbone_raw_array = \
            process_seq(backbone_seq, args, isbackbone=True)
        query_seq_names, query_seq_names_raw, query_seq_tensor, query_raw_array = \
            process_seq(query_seq, args, isbackbone=False)
    else:
        # breakpoint()
        if not (recon_model is None):
            if (args.recon_backbone_emb is None) or (args.backbone_id is None) or (args.backbone_gap is None):
                backbone_seq_names, backbone_seq_tensor, backbone_mask = process_seq(backbone_seq, args,
                                                                                     isbackbone=True, need_mask=True)
            else:
                backbone_seq_names = torch.load(args.backbone_id)
                backbone_mask = torch.load(args.backbone_gap)
            query_seq_names, query_seq_tensor, query_mask = process_seq(query_seq, args, isbackbone=False,
                                                                        need_mask=True)
        else:
            if (args.backbone_emb is None) or (args.backbone_id is None):
                backbone_seq_names, backbone_seq_tensor = process_seq(backbone_seq, args, isbackbone=True)
            else:
                backbone_seq_names = torch.load(args.backbone_id)
            query_seq_names, query_seq_tensor = process_seq(query_seq, args, isbackbone=False)

    for param in model.parameters():
        param.requires_grad = False
    print('finish data processing!')
    print(f'{len(backbone_seq_names)} backbone sequences')
    print(f'{len(query_seq_names)} query sequence(s)')
    print(f'calculating embeddings...')
    if (args.backbone_emb is None) or (args.backbone_id is None):
        backbone_encodings = get_embeddings(backbone_seq_tensor, model)
    else:
        backbone_encodings = torch.load(args.backbone_emb)
    query_encodings = get_embeddings(query_seq_tensor, model)
    # torch.save(query_encodings, f'{dis_file_root}/query_embeddings.pt')
    # torch.save(query_seq_names, f'{dis_file_root}/query_names.pt')
    torch.save(backbone_encodings, f'{dis_file_root}/backbone_embeddings.pt')
    torch.save(backbone_seq_names, f'{dis_file_root}/backbone_names.pt')

    if not (recon_model is None):
        if (args.recon_backbone_emb is None) or (args.backbone_id is None) or (args.backbone_gap is None):
            recon_backbone_encodings = get_embeddings(backbone_seq_tensor, recon_model, backbone_mask)
        else:
            recon_backbone_encodings = torch.load(args.recon_backbone_emb)
        recon_query_encodings = get_embeddings(query_seq_tensor, recon_model, query_mask)
        torch.save(recon_backbone_encodings, f'{dis_file_root}/recon_backbone_embeddings.pt')

    print(f'finish embedding calculation!')
    print(f'calculating distance matrix...')
    t2 = time.time()
    # print('calculate embeddings', t2 - t1)

    # query_dist = distance(query_encodings, backbone_encodings, args.distance_mode) * args.distance_ratio
    query_dist = distance(query_encodings, backbone_encodings, args.distance_mode) * args.distance_ratio
    if 'square_root' in args.weighted_method:
        query_dist = query_dist ** 2

    if recon_model:
        gap_portion = 1 - query_mask.int().sum(-1) / query_mask.shape[-1]
        recon_query_dist = distance(recon_query_encodings, recon_backbone_encodings,
                                    args.distance_mode) * args.distance_ratio
        if 'square_root' in args.weighted_method:
            recon_query_dist = recon_query_dist ** 2
        query_dist = query_dist * (1 - gap_portion) + recon_query_dist * gap_portion

    t3 = time.time()
    # print('calculate distance', t3 - t2)
    query_dist = np.array(query_dist)
    query_dist[query_dist < 1e-3] = 0
    data_origin = dict(zip(query_seq_names, list(query_dist.astype(str))))
    data_origin = "\t" + "\t".join(backbone_seq_names) + "\n" + \
                  "\n".join([str(k) + "\t" + "\t".join(data_origin[k]) for k in data_origin]) + "\n"
    t4 = time.time()
    # print('convert string', t4 - t3)
    with open(os.path.join(dis_file_root, f'depp.csv'), 'w') as f:
        f.write(data_origin)
    t5 = time.time()
    # print('save string', t5 - t4)
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
    print("take {:.2f} seconds".format(t5 - t1))


def save_repr_emb(model, args):
    t1 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    print('calculating backbone embeddings...')

    def get_backbone_embeddings(i):
        print(f'embedding {i} cluster...')
        backbone_seq_file = f"{args.seqdir}/{i}.fa"
        backbone_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
        backbone_seq_names, backbone_seq_tensor = process_seq(backbone_seq, args, isbackbone=True)
        backbone_seq_tensor_idx = np.random.choice(len(backbone_seq_tensor), size=100, replace=False)
        backbone_seq_tensor = backbone_seq_tensor[backbone_seq_tensor_idx]
        backbone_encodings = {i: get_embeddings(backbone_seq_tensor, model, query=False, model_idx=i).mean(0) for i in
                              range(model.hparams.cluster_num)}
        return backbone_encodings

    backbone_embs_tmp = [get_backbone_embeddings(i) for i in range(model.hparams.cluster_num)]

    backbone_encodings_dict = {i: backbone_embs_tmp[i] for i in range(model.hparams.cluster_num)}

    torch.save(backbone_encodings_dict, f'{args.outdir}/repr_emb.pt')
    t2 = time.time()
    print('finish! take {:.2f} seconds.'.format(t2 - t1))

def p_q_cal(tree, group):
    for node in tree.traverse_postorder():
        if node.is_leaf():
            if node.label in group:
                node.group_cnt = {group[node.label]: 1}
            else:
                node.group_cnt = {}
            node.child_sum = 1
        else:
            node.group_cnt = {}
            node.child_sum = 0
            for child in node.child_nodes():
                for g in child.group_cnt:
                    node.group_cnt[g] = node.group_cnt.get(g, 0) + child.group_cnt[g]
                node.child_sum += child.child_sum

    for node in tree.traverse_preorder():
        node.score = {g: node.group_cnt[g] ** 2 / (tree.root.group_cnt[g] * node.child_sum) for g in node.group_cnt}

    best_nodes = {}
    for node in tree.traverse_postorder():
        for g in node.score:
            if g not in best_nodes or best_nodes[g][1] < node.score[g]:
                best_nodes[g] = (node, node.score[g])
    return best_nodes
        # node.q = {g: node.group_cnt[g] / tree.child_sum for g in node.group_cnt}

def custom_clustering(distance_matrix, threshold):
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='complete')

    # Cluster the data based on the threshold
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

    return clusters

def hamming_distance(seq1, seq2):
    non_gap_mask = ~(np.all(seq1 == 0.25, axis=-2) | np.all(seq2 == 0.25, axis=-2))
    # non_gap_mask = np.all((seq1 != 0.25) & (seq2 != 0.25), axis=-2)
    return np.sum(np.any(seq1 != seq2, axis=-2) * non_gap_mask, axis=-1).astype(float) / non_gap_mask.sum(-1)

def calculate_distance_matrix(seq1, seq2):
    # num_species1 = len(seq1)
    # num_species2 = len(seq2)
    # seqs = np.array(list(one_hot_sequences.values()))
    tiled_seqs1 = seq1[:, np.newaxis, :, :]
    tiled_seqs2 = seq2[np.newaxis, :, :]
    distance_matrix = hamming_distance(tiled_seqs1, tiled_seqs2)
    return distance_matrix

def combine_identical_seq_intree(tree, seq, thr=None, outdir=None, cluster=None, replicate=False):
    if cluster is not None:
        group = pd.read_csv(cluster, sep='\t', index_col=0)
        if replicate:
            group = group.groupby(group.index.str.split('_').str[0]).agg(lambda x: x.mode()[0])
            group.index = [i.split('_')[0] for i in group.index]
            group.loc[group['ClusterNumber'] == -1, 'ClusterNumber'] = np.arange((group['ClusterNumber'] == -1).sum()) + group['ClusterNumber'].max() + 1
            group = {i: group.loc[i]['ClusterNumber'] for i in group.index}
    elif thr is None:
        unique_seq = set([seq[i].tobytes() for i in seq])
        label_seq = dict(zip(unique_seq, np.arange(len(unique_seq))))
        group = {s: label_seq[seq[s].tobytes()] for i, s in enumerate(seq)}
    else:
        distance_matrix = np.zeros([len(seq), len(seq)])
        one_hot_seq = np.stack(list(seq.values()), axis=0)
        for i in range(0, len(distance_matrix), 100):
            print(i)
            distance_matrix[i: i+100] = calculate_distance_matrix(one_hot_seq[i: i+100], one_hot_seq)
        clusters = custom_clustering(distance_matrix, threshold=thr)
        group = {s: clusters[i] for i, s in enumerate(seq)}

    best_node = p_q_cal(tree, group=group)
    best_node_group = collections.defaultdict(list)
    LCA_node_name = {}
    for s in group.keys():
        best_node_group[best_node[group[s]][0]].append(s)
        if best_node[group[s]][0].is_leaf():
            LCA_node_name[s] = s
        else:
            LCA_node_name[s] = f'DEPP-LCA{group[s]}'

    name_to_node = {i.label: i for i in tree.traverse_leaves()}

    # for leaf in tree.traverse_leaves():
    #     if leaf.label not in group:
    #         continue
    #     if not best_node[group[leaf.label]][0].is_leaf():
    #         leaf.label = f"OriginalNode{leaf.label}"

    for node in best_node_group:
        if node.is_leaf():
            continue
        # edge_length = np.mean([node.distance[i] for i in best_node_group[node]])
        lca_nodes = np.unique([LCA_node_name[item] for item in best_node_group[node]])
        for lca in lca_nodes:
            edge_length = np.mean([tree.distance_between(name_to_node[i], node) for i in best_node_group[node] if LCA_node_name[i] == lca])
            new_node = ts.Node(edge_length=edge_length, label=LCA_node_name[best_node_group[node][0]])
            node.add_child(new_node)
        # for s in best_node_group[node]:
        #     new_node.add_child(ts.Node(label=s, edge_length=0))

    # for node in tree.traverse_postorder():
    #     node.distance = None

    if outdir is not None:
        tree.write_tree_newick(f'{outdir}/tree_forplacement.newick')

    for leaf in tree.traverse_leaves():
        if (leaf.label in group) and (LCA_node_name[leaf.label] != leaf.label):
            leaf.parent.remove_child(leaf)

    modified = True
    while modified:
        modified = False
        for node in tree.traverse_leaves():
            if (node.label not in group) and ("DEPP-LCA" not in node.label):
                node.parent.remove_child(node)
                modified = True

    tree.suppress_unifurcations()
    # tree.resolve_polytomies()
    return tree, LCA_node_name

# def save_depp_dist(model, args, recon_model=None):
#     t1 = time.time()
#     if model is not None:
#         model.eval()
#         args.replicate_seq = model.hparams.replicate_seq
#         args.distance_ratio = model.hparams.distance_ratio
#         args.gap_encode = model.hparams.gap_encode
#         args.jc_correct = model.hparams.jc_correct
#     elif recon_model is not None:
#         args.replicate_seq = recon_model.hparams.replicate_seq
#         args.distance_ratio = recon_model.hparams.distance_ratio
#         args.gap_encode = recon_model.hparams.gap_encode
#         args.jc_correct = recon_model.hparams.jc_correct
#
#     print('processing data...')
#     backbone_seq_file = args.backbone_seq_file
#     query_seq_file = args.query_seq_file
#     dis_file_root = os.path.join(args.outdir)
#     # args.distance_ratio = float(1.0 / float(args.embedding_size) / 10 * float(args.distance_alpha))
#     #args.replicate_seq = model.hparams.replicate_seq
#     print('jc_correct', args.jc_correct)
#     if args.jc_correct:
#         args.jc_ratio = model.hparams.jc_ratio
#     if not os.path.exists(dis_file_root):
#         os.makedirs(dis_file_root, exist_ok=True)
#
#     backbone_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
#     query_seq = SeqIO.to_dict(SeqIO.parse(query_seq_file, "fasta"))
#
#     if args.jc_correct:
#         backbone_seq_names, backbone_seq_names_raw, backbone_seq_tensor, backbone_raw_array = \
#             process_seq(backbone_seq, args, isbackbone=True)
#         query_seq_names, query_seq_names_raw, query_seq_tensor, query_raw_array = \
#             process_seq(query_seq, args, isbackbone=False)
#     else:
#         # breakpoint()
#         if not (recon_model is None):
#             if (args.recon_backbone_emb is None) or (args.backbone_id is None) or (args.backbone_gap is None):
#                 backbone_seq_names, backbone_seq_tensor, backbone_mask = process_seq(backbone_seq, args, isbackbone=True, need_mask=True)
#                 torch.save(backbone_mask, f'{dis_file_root}/backbone_gap.pt')
#             else:
#                 backbone_seq_names = torch.load(args.backbone_id)
#                 backbone_mask = torch.load(args.backbone_gap)
#             query_seq_names, query_seq_tensor, query_mask = process_seq(query_seq, args, isbackbone=False, need_mask=True)
#         else:
#             if (args.backbone_emb is None) or (args.backbone_id is None):
#                 backbone_seq_names, backbone_seq_tensor = process_seq(backbone_seq, args, isbackbone=True)
#             else:
#                 backbone_seq_names = torch.load(args.backbone_id)
#             query_seq_names, query_seq_tensor = process_seq(query_seq, args, isbackbone=False)
#     if model is not None:
#         for param in model.parameters():
#             param.requires_grad = False
#     if recon_model is not None:
#         for param in recon_model.parameters():
#             param.requires_grad = False
#     print('finish data processing!')
#     print(f'{len(backbone_seq_names)} backbone sequences')
#     print(f'{len(query_seq_names)} query sequence(s)')
#     print(f'calculating embeddings...')
#     if not (model is None):
#         if (args.backbone_emb is None) or (args.backbone_id is None):
#             backbone_encodings = get_embeddings(backbone_seq_tensor, model)
#         else:
#             backbone_encodings = torch.load(args.backbone_emb)
#         query_encodings = get_embeddings(query_seq_tensor, model)
#     #torch.save(query_encodings, f'{dis_file_root}/query_embeddings.pt')
#     #torch.save(query_seq_names, f'{dis_file_root}/query_names.pt')
#     #torch.save(backbone_encodings, f'{dis_file_root}/backbone_embeddings.pt')
#     #torch.save(backbone_seq_names, f'{dis_file_root}/backbone_names.pt')
#
#     if not (recon_model is None):
#         if (args.recon_backbone_emb is None) or (args.backbone_id is None) or (args.backbone_gap is None):
#             recon_backbone_encodings = get_embeddings(backbone_seq_tensor, recon_model, backbone_mask)
#         else:
#             recon_backbone_encodings = torch.load(args.recon_backbone_emb)
#         recon_query_encodings = get_embeddings(query_seq_tensor, recon_model, query_mask)
#         torch.save(recon_backbone_encodings, f'{dis_file_root}/recon_backbone_embeddings.pt')
#
#     print(f'finish embedding calculation!')
#     print(f'calculating distance matrix...')
#     t2 = time.time()
#     #print('calculate embeddings', t2 - t1)
#
#     # query_dist = distance(query_encodings, backbone_encodings, args.distance_mode) * args.distance_ratio
#     if model:
#         query_dist = distance(query_encodings, backbone_encodings, args.distance_mode) * args.distance_ratio
#         if 'square_root' in args.weighted_method:
#             query_dist = query_dist ** 2
#
#     if recon_model:
#         gap_portion = 1 - query_mask.int().sum(-1) / query_mask.shape[-1]
#         recon_query_dist = distance(recon_query_encodings, recon_backbone_encodings, args.distance_mode) * args.distance_ratio
#         if 'square_root' in args.weighted_method:
#             recon_query_dist = recon_query_dist ** 2
#         if model:
#             query_dist = query_dist * (1 - gap_portion) + recon_query_dist * gap_portion
#         else:
#             query_dist = recon_query_dist
#
#     t3 = time.time()
#     #print('calculate distance', t3 - t2)
#     query_dist = np.array(query_dist)
#     query_dist[query_dist < 1e-3] = 0
#     data_origin = dict(zip(query_seq_names, list(query_dist.astype(str))))
#     data_origin = "\t" + "\t".join(backbone_seq_names) + "\n" + \
#                   "\n".join([str(k) + "\t"+ "\t".join(data_origin[k]) for k in data_origin]) + "\n"
#     t4 = time.time()
#     #print('convert string', t4 - t3)
#     with open(os.path.join(dis_file_root, f'depp.csv'), 'w') as f:
#         f.write(data_origin)
#     t5 = time.time()
#     #print('save string', t5 - t4)
#     # data_origin = pd.DataFrame.from_dict(data_origin, orient='index', columns=backbone_seq_names)
#
#     if args.query_dist:
#         idx = data_origin.index
#         data_origin = data_origin[idx]
#     # data_origin.to_csv(os.path.join(dis_file_root, f'depp.csv'), sep='\t')
#     # if not os.path.isdir(f'{args.outdir}/depp_tmp'):
#     #     os.makedirs(f'{args.outdir}/depp_tmp')
#     # with open(f'{args.outdir}/depp_tmp/seq_name.txt', 'w') as f:
#     #     f.write("\n".join(query_seq_names) + '\n')
#     print('original distanace matrix saved!')
#     print("take {:.2f} seconds".format(t5-t1))
