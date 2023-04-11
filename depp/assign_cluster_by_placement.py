#!/usr/bin/env python3
import collections
import json
import argparse
import treeswift as ts
import pandas as pd

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--tree', type=str)
parser.add_argument('--treecluster', type=str)
parser.add_argument('--query', type=str)
parser.add_argument('--outdir', type=str)
args = parser.parse_args()

tree = ts.read_tree_newick(args.tree)
cluster = pd.read_csv(args.treecluster, sep='\t', index_col=0)

query_class = collections.defaultdict(list)

with open(args.query, 'r') as f:
    query = f.read().split('\n')
    while query[-1] == '':
        query = query[:-1]
    query = set(query)

for n in tree.traverse_postorder():
    if n.is_leaf():
        if n.label in query:
            n.color = ('query', [n.label])
        else:
            n.color = (cluster.loc[n.label]['ClusterNumber'].item() - 1, 0)
    else:
        query_cnt = []
        color_cnt = {}
        for c in n.child_nodes():
            color = c.color
            if color[0] == 'query':
                query_cnt += color[1]
            else:
                color_cnt[color[0]] = min(color_cnt.get(color[0], float('inf')), color[1] + c.edge_length)
        if len(color_cnt) == 0:
            n.color = ('query', query_cnt)
        else:
            color = sorted(color_cnt.items(), key=lambda x: x[1])[0]
            n.color = color
            for q in query_cnt:
                query_class[color[0]].append(q)

for k in query_class.keys():
    with open(f'{args.outdir}/{k}.txt', 'w') as f:
        f.write("\n".join([str(i) for i in query_class[k]]) + '\n')