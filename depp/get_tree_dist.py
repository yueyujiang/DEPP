#!/usr/bin/env python3

import argparse
import treeswift as ts
import pandas as pd

parser = argparse.ArgumentParser(description='get the corresponding tree distance matrix of input distance')
parser.add_argument('--treefile', type=str)
parser.add_argument('--distfile', type=str, default=None)
parser.add_argument('--outfile', type=str)
args = parser.parse_args()

dist = ts.read_tree_newick(args.treefile).distance_matrix(leaf_labels=True)
for k in dist:
    dist[k][k] = 0

if args.distfile:
    odist = pd.read_csv(args.distfile, sep='\t', index_col=0)
    for i in odist.index:
        odist.loc[i, :] = [dist[str(i)][str(j)] for j in odist.keys()]
else:
    keys = dist.keys()
    odist = {}
    for k in keys:
        odist[k] = [dist[k][i] for i in keys]
    odist = pd.DataFrame.from_dict(odist, orient='index', columns=keys)

odist.to_csv(args.outfile, sep='\t')
