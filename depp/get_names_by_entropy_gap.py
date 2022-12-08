#!/usr/bin/env python3

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--gapfile', type=str)
parser.add_argument('--entropyfile', type=str)
parser.add_argument('--entropy', type=float, default=float('inf'))
parser.add_argument('--gap', type=float, default=float('inf'))
parser.add_argument('--outfile', type=str)
args = parser.parse_args()

a = pd.read_csv(args.gapfile, sep='\t', index_col=0, dtype={0:'str',1:'float'}, names=['id', 'gap'])
b = pd.read_csv(args.entropyfile, sep='\t', index_col=0, dtype={0:'str',1:'float'}, names=['id', 'entropy'])
a = pd.concat([a, b], axis=1)

names = a[(a['entropy']<args.entropy)&(a['gap']<args.gap)].index

with open(args.outfile, 'w') as f:
    f.write("\n".join(names)+'\n')
