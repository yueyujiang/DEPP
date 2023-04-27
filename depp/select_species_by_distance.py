#!/usr/bin/env python3

import pandas as pd
import os
import numpy as np
import collections
from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--distlist', nargs='+')
parser.add_argument('--cluster', type=str)
parser.add_argument('--outfile', type=str)
args = parser.parse_args()

with open(args.cluster, 'r') as f:
    cluster = f.read().split('\n')
    while cluster[-1] == '':
        cluster = cluster[:-1]

dist = []
for d in args.distlist:
    dist.append(pd.read_csv(d, sep='\t', index_col=0)[cluster])
dist = pd.concat(dist, axis=0)

def select_species(dist):
    dist = dist.loc[np.isin(dist.index, cluster, invert=True)]
    d = dist.min(axis=1)
    add_repr = d.index[np.argsort(d).values[:4000]]
    return add_repr


with open(args.outfile, 'w') as f:
    f.write("\n".join([str(i) for i in select_species(dist)])+'\n')
