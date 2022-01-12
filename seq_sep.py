#!/usr/bin/env python3
import argparse
from Bio import SeqIO

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--infile', type=str)
parser.add_argument('--outdir', type=str)
args = parser.parse_args()

all_seq = SeqIO.to_dict(SeqIO.parse(args.infile, "fasta"))

v4_100 = [f'>{i}\n{str(all_seq[i].seq)}\n' for i in all_seq if len(all_seq[i].seq) <= 130]
v4_150 = [f'>{i}\n{str(all_seq[i].seq)}\n' for i in all_seq if 130 < len(all_seq[i].seq) <= 240]
v4 = [f'>{i}\n{str(all_seq[i].seq)}\n' for i in all_seq if 240 < len(all_seq[i].seq) <= 400]
v3_v4 = [f'>{i}\n{str(all_seq[i].seq)}\n' for i in all_seq if 400 < len(all_seq[i].seq) <= 1200]
full_length = [f'>{i}\n{str(all_seq[i].seq)}\n' for i in all_seq if len(all_seq[i].seq) > 1200]

if len(v4_100) != 0:
    with open(f'{args.outdir}/16s_v4_100.fa', 'w') as f:
        f.write("".join(v4_100)+'\n')

if len(v4_150) != 0:
    with open(f'{args.outdir}/16s_v4_150.fa', 'w') as f:
        f.write("".join(v4_150)+'\n')

if len(v4) != 0:
    with open(f'{args.outdir}/16s_v4.fa', 'w') as f:
        f.write("".join(v4)+'\n')

if len(v3_v4) != 0:
    with open(f'{args.outdir}/16s_v3_v4.fa', 'w') as f:
        f.write("".join(v3_v4)+'\n')

if len(full_length) != 0:
    with open(f'{args.outdir}/16s_full_length.fa', 'w') as f:
        f.write("".join(full_length)+'\n')
