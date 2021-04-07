#!/usr/bin/python

import pandas as pd
import collections
import os
import numpy as np
import argparse
import dendropy
import copy
import os
import re
from Bio import SeqIO

parser = argparse.ArgumentParser(description='test')
parser.add_argument('-f', '--filter_thr', type=str)
parser.add_argument('-o', '--out-dir', type=str)
parser.add_argument('-g', '--gene', type=str)
args = parser.parse_args()

thr = args.filter_thr
infile = f'{args.out_dir}/{args.gene}_aligned.fa'

seq1 = SeqIO.to_dict(SeqIO.parse(args.infile, "fasta"))

long = ""
for a in seq1:
    s = list(seq1[a])
    t = np.array(s)
    if len(t[t!='-']) > thr:
        long += f'>{a}\n'
        long += f'{"".join(s)}\n'
with open(f'{args.out_dir}/{args.gene}_aligned_filtered.fa', 'w') as f:
    f.write(long)


