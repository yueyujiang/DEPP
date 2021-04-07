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
parser.add_argument('-o', '--output-dir', type=str)
parser.add_argument('-a', '--accessory-dir', type=str)
parser.add_argument('-p', '--prefix', type=str)
args = parser.parse_args()

query_dir = args.output_dir
accessory_dir = args.accessory_dir
prefix = args.prefix

tree = dendropy.Tree.get(path=f'{accessory_dir}/wol.nwk', schema='newick')
backbone_leaves = [a.taxon.label for a in tree.leaf_nodes()]
dis_mat = pd.DataFrame(index=None, columns=backbone_leaves)
weights = {}
types = 'bins'
for i, d in enumerate(os.listdir(query_dir)):
    if not d.startswith(prefix):
        continue
    file = f'{query_dir}/{d}/depp_correction.csv'
    if not os.path.isfile(file):
        continue
    a = pd.read_csv(file, sep='\t')
    a = a.set_index('Unnamed: 0')
    c = a.applymap(lambda x: [x])
    add_l = [i for i in backbone_leaves if i not in c]
    c = pd.concat([c, pd.DataFrame(columns=add_l)])
    index = c.index.union(dis_mat.index)
    c = c.reindex(index).applymap(lambda x: x if isinstance(x, list) else [])
    dis_mat = dis_mat.reindex(index).applymap(lambda x: x if isinstance(x, list) else [])
    dis_mat = c + dis_mat

import numpy as np
def agg_cell(x):
    # print(x)
    l = len(x)
    if l == 0:
        return -1
    if l <= 3:
        return np.median(x)
    return np.mean(sorted(x)[l//4: int(l/4*3)])

dis_mat = dis_mat.applymap(agg_cell)
if not os.path.isdir(f'{query_dir}/summary/'):
    os.makedirs(f'{query_dir}/summary/')
if prefix == 'p':
    dis_mat.to_csv(f'{query_dir}/summary/marker_genes_dist.csv', sep='\t')
# print(sorted(lens)[-4000000: -3990000])
#             weights[k1][k2].append(1-qs[d])
# print(sorted(lens)[-4000000: -3990000])