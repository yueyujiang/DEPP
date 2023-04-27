#!/usr/bin/env python3

import argparse
import collections
from Bio import SeqIO

# this is for sequences with multiple copies

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--infile', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--name-list', nargs='+')
args = parser.parse_args()

seq = SeqIO.to_dict(SeqIO.parse(args.infile, "fasta"))

mapping = collections.defaultdict(list)
for s in seq:
    mapping[s.split('_')[0]].append(s)

for name_file in args.name_list:
    with open(name_file, 'r') as f:
        seqname = f.read().split('\n')
        if seqname[-1] == '':
            seqname = seqname[:-1]
    s = ""
    for n in seqname:
        if n in mapping:
            for t in mapping[n]:
                s += f">{t}\n{str(seq[t].seq)}\n"
    with open(f"{args.outdir}/{name_file.split('.')[0].split('/')[-1]}.fa", "w") as f:
        f.write(s)
