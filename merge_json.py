#!/usr/bin/python

import json
import argparse
import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--out-dir', type=str)
args = parser.parse_args()

output_dir = f'{args.out_dir}/summary'
# print(output_dir)
asvs = set([a.split('_placement')[0] for a in os.listdir(output_dir) if a.startswith('16')])

for i, asv in enumerate(asvs):
    with open(f'{output_dir}/{asv}_placement.jplace', 'r') as f:
        tree_tmp = json.load(f)
        for k in range(len(tree_tmp['placements'])):
            tree_tmp['placements'][k]['n'][0] = f'{asv}_' + tree_tmp['placements'][k]['n'][0]
        if i == 0:
            tree1 = tree_tmp
        else:
            tree1['placements'] += tree_tmp['placements']

with open(f'{output_dir}/marker_genes_placement.jplace', 'r') as f:
    tree2 = json.load(f)
    for i in range(len(tree2['placements'])):
        tree2['placements'][i]['n'][0] = 'MAG_' + tree2['placements'][i]['n'][0]
tree1['placements'] += tree2['placements']

with open(f'{output_dir}/mag_asv.jplace', 'w') as f:
    json.dump(tree1, f)