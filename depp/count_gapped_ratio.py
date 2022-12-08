#!/usr/bin/env python3

import argparse
from Bio import SeqIO

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--infile', type=str)
parser.add_argument('--outfile', type=str)
args = parser.parse_args()

all_seq = SeqIO.to_dict(SeqIO.parse(args.infile, "fasta"))
L = len(list(all_seq.values())[0])

def cnt_gap(s):
    return len([i for i in s if i in ['-', 'N']]) / len(s)

s = ""
for i in all_seq:
    s += f"{i}\t{cnt_gap(str(all_seq[i].seq))}\n"

with open(args.outfile, "w") as f:
    f.write(s)
