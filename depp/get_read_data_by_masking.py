#!/usr/bin/env python3

import argparse
from Bio import SeqIO
import numpy as np

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--backbone-seq', type=str)
parser.add_argument('--query-seq', type=str)
parser.add_argument('--read-region', type=str)
parser.add_argument('--outdir', type=str)
args = parser.parse_args()

backbone_seq = SeqIO.to_dict(SeqIO.parse(args.backbone_seq, "fasta"))
query_seq = SeqIO.to_dict(SeqIO.parse(args.query_seq, "fasta"))

backbone_seq = np.stack([np.array(list(backbone_seq[i].seq)) for i in backbone_seq], axis=0)
query_seq = np.stack([np.array(list(query_seq[i].seq)) for i in query_seq], axis=0)

read_region = args.read_region
if args.read_region != "noregion":
    start, end = read_region.split(':')
    start, end = int(start)-1, int(end)
    print(f'Read region is specified')
else:
    non_gap_ratio = (query_seq == '-').sum(axis=0) / len(query_seq)
    non_gappy = np.arange(query_seq.shape[-1])[non_gap_ratio>0.6]
    start, end = non_gappy.min(), non_gappy.max()
    print('No read region is specified')
print(f'Starting position and ending position is site {start} and site {end} respectively')
backbone_seq = backbone_seq[:start] = '-'
query_seq = query_seq[end:] = '-'
backbone_seq = [f'>{i}\n{"".join(backbone_seq[i])}\n' for i in backbone_seq]
query_seq = [f'>{i}\n{"".join(query_seq[i])}\n' for i in query_seq]

with open(f"{args.outdir}/backbone.fa", 'w') as f:
    f.write("".join(backbone_seq))

with open(f"{args.outdir}/query.fa", 'w') as f:
    f.write("".join(query_seq))
