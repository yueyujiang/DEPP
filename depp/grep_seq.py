#!/usr/bin/env python3

import argparse
from Bio import SeqIO

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--infile', type=str)
parser.add_argument('--outfile', type=str)
parser.add_argument('--seqnames', type=str)
parser.add_argument('--replicate', action='store_true')
parser.add_argument('--inverse', action='store_true')
args = parser.parse_args()

with open(args.seqnames, 'r') as f:
    seqname = set(f.read().split('\n'))

seq = SeqIO.to_dict(SeqIO.parse(args.infile, "fasta"))

if not args.inverse:
    if args.replicate:
        seq = [f'>{i}\n{str(seq[i].seq)}\n' for i in seq if i.split('_')[0] in seqname]
    else:
        seq = [f'>{i}\n{str(seq[i].seq)}\n' for i in seq if i in seqname]
else:
    if args.replicate:
        seq = [f'>{i}\n{str(seq[i].seq)}\n' for i in seq if i.split('_')[0] not in seqname]
    else:
        seq = [f'>{i}\n{str(seq[i].seq)}\n' for i in seq if i not in seqname]

with open(args.outfile, 'w') as f:
    f.write("".join(seq)+'\n')
