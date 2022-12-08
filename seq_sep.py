#!/usr/bin/env python3
import argparse
from Bio import SeqIO

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--infile', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--aligned', action='store_true')
args = parser.parse_args()

all_seq = SeqIO.to_dict(SeqIO.parse(args.infile, "fasta"))

if not args.aligned:
    v4 = [f'>{i}\n{str(all_seq[i].seq)}\n' for i in all_seq if len(all_seq[i].seq) <= 500]
    full_length = [f'>{i}\n{str(all_seq[i].seq)}\n' for i in all_seq if len(all_seq[i].seq) > 500]

    if len(v4) != 0:
        with open(f'{args.outdir}/16s_v4.fa', 'w') as f:
            f.write("".join(v4)+'\n')

    if len(full_length) != 0:
        with open(f'{args.outdir}/16s_full_length.fa', 'w') as f:
            f.write("".join(full_length)+'\n')
else:
    def get_portion(s, start, end):
        s_r = s[start: end+1].replace('-', '')
        return len(s_r) / len(s[start: end+1])
    v4_100_p = {i: get_portion(str(all_seq[i].seq), 0, 98) for i in all_seq}
    v4_150_p = {i: get_portion(str(all_seq[i].seq), 0, 148) for i in all_seq}
    v4_p = {i: get_portion(str(all_seq[i].seq), 0, 248) for i in all_seq}
    def v4_condition(i): return (v4_p[i] >= 0.8) or (v4_p[i] >= v4_150_p[i] and v4_p[i] >= v4_100_p[i])
    def v4_150_condition(i): return (not v4_condition(i)) and (v4_150_p[i] >= 0.8 or v4_150_p[i] >= v4_100_p[i])
    def v4_100_condition(i): return (not v4_condition(i)) and (not v4_150_condition(i))
    v4 = [f'>{i}\n{str(all_seq[i].seq)}\n' for i in all_seq if v4_condition(i)]
    v4_150 = [f'>{i}\n{str(all_seq[i].seq)[:149]}\n' for i in all_seq if v4_150_condition(i)]
    v4_100 = [f'>{i}\n{str(all_seq[i].seq)[:99]}\n' for i in all_seq if v4_100_condition(i)]

    if len(v4) != 0:
        with open(f'{args.outdir}/16s_v4_aligned.fa', 'w') as f:
            f.write("".join(v4)+'\n')

    if len(v4_150) != 0:
        with open(f'{args.outdir}/16s_v4_150_aligned.fa', 'w') as f:
            f.write("".join(v4_150)+'\n')

    if len(v4_100) != 0:
        with open(f'{args.outdir}/16s_v4_100_aligned.fa', 'w') as f:
            f.write("".join(v4_100)+'\n')