import torch
import os
import collections
import pandas as pd
import math
import numpy as np
from Bio import SeqIO

def get_seq_length(args):
    backbone_seq_file = args.backbone_seq_file
    seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
    args.sequence_length = len(list(seq.values())[0])

def distance(nodes1, nodes2, mode):
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
        return torch.sum((nodes1 - nodes2) ** 2, dim=-1)
    elif mode == 'L1':
        return torch.sum(abs(nodes1 - nodes2), dim=-1)
    elif mode == 'cosine':
        return 1 - torch.nn.functional.cosine_similarity(nodes1, nodes2, dim=-1)
    elif mode == 'tan':
        cosine = torch.nn.functional.cosine_similarity(nodes1, nodes2, dim=-1)
        return (1 - cosine ** 2) / (cosine + 1e-9)


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


def save_depp_dist(model, args):
    print('processing data...')
    backbone_seq_file = args.backbone_seq_file
    query_seq_file = args.query_seq_file
    dis_file_root = os.path.join(args.outdir)
    args.distance_ratio = float(1.0 / float(args.embedding_size) / 10 * float(args.distance_alpha))
    if not os.path.exists(dis_file_root):
        os.makedirs(dis_file_root)

    backbone_seq = SeqIO.to_dict(SeqIO.parse(backbone_seq_file, "fasta"))
    query_seq = SeqIO.to_dict(SeqIO.parse(query_seq_file, "fasta"))

    backbone_seq_tensor = []
    backbone_seq_names = []
    L = len(list(backbone_seq.values())[0])
    for k in backbone_seq:
        seq = np.zeros([4, L])
        raw_seq = np.array(backbone_seq[k])
        seq[0][raw_seq == 'A'] = 1
        seq[1][raw_seq == 'C'] = 1
        seq[2][raw_seq == 'G'] = 1
        seq[3][raw_seq == 'T'] = 1
        backbone_seq_tensor.append(torch.from_numpy(seq).unsqueeze(0))
        backbone_seq_names.append(k)
    backbone_seq_tensor = torch.cat(backbone_seq_tensor, dim=0)

    query_seq_tensor = []
    query_seq_names = []
    for k in query_seq:
        seq = np.zeros([4, L])
        raw_seq = np.array(query_seq[k])
        seq[0][raw_seq == 'A'] = 1
        seq[1][raw_seq == 'C'] = 1
        seq[2][raw_seq == 'G'] = 1
        seq[3][raw_seq == 'T'] = 1
        query_seq_tensor.append(torch.from_numpy(seq).unsqueeze(0))
        query_seq_names.append(k)
    query_seq_tensor = torch.cat(query_seq_tensor, dim=0)

    for param in model.parameters():
        param.requires_grad = False
    print('finish data processing!')
    print(f'{len(backbone_seq_names)} backbone sequences')
    print(f'{len(query_seq_names)} query sequence(s)')
    print(f'calculating embeddings...')
    backbone_encodings = []
    for i in range(math.ceil(len(backbone_seq_tensor) / 10.0)):
        encodings_tmp = model(backbone_seq_tensor[i * 10: (i + 1) * 10].float()).detach()
        backbone_encodings.append(encodings_tmp)
    backbone_encodings = torch.cat(backbone_encodings, dim=0)

    query_encodings = []
    for i in range(math.ceil(len(query_seq_tensor) / 10.0)):
        encodings_tmp = model(query_seq_tensor[i * 10: (i + 1) * 10].float()).detach()
        query_encodings.append(encodings_tmp)
    query_encodings = torch.cat(query_encodings, dim=0)
    print(f'finish embedding calculation!')

    backbone_dist = []
    for i in range(len(backbone_encodings)):
        backbone_dist.append(distance(backbone_encodings[i], backbone_encodings, args.distance_mode) * args.distance_ratio)
    backbone_dist = torch.cat(backbone_dist, dim=0)
    median_max_distance = float(torch.median((backbone_dist ** 2).max(-1)[0]))

    print(f'median_max: {median_max_distance}')

    query_dist = distance(query_encodings, backbone_encodings, args.distance_mode) * args.distance_ratio

    data_origin = collections.defaultdict(list)
    data_median = collections.defaultdict(list)
    for j, qe in enumerate(query_encodings):
        for i, e in enumerate(backbone_encodings):
            d = float(query_dist[j, i])
            if args.weighted_method == 'square_root_fm':
                d = d ** 2
            # print(d)
            data_origin[query_seq_names[j]].append(d)
            data_median[query_seq_names[j]].append(d)

        max_model_distance = max(data_origin[query_seq_names[j]])
        r_median = median_max_distance / max_model_distance
        for m, v in enumerate(data_median[query_seq_names[j]]):
            data_median[query_seq_names[j]][m] = data_median[query_seq_names[j]][m] * r_median

    data_median = pd.DataFrame.from_dict(data_median, orient='index', columns=backbone_seq_names)
    data_origin = pd.DataFrame.from_dict(data_origin, orient='index', columns=backbone_seq_names)

    data_median.to_csv(os.path.join(dis_file_root, f'depp_correction.csv'), sep='\t')
    data_origin.to_csv(os.path.join(dis_file_root, f'depp.csv'), sep='\t')

    print('distanace matrix saved!')
#
# def true_distance(nodes1, nodes2, distance_matrix):
#     gt_distance_list = itemgetter(*nodes1)(distance_matrix)
#     gt_distance = [torch.tensor(itemgetter(*nodes2)(item)).view(1, len(nodes2)) for item in gt_distance_list]
#     gt_distance = torch.cat(gt_distance, dim=0)
#     return gt_distance