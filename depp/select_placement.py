#!/usr/bin/env python3
import collections
import json
import argparse
import treeswift as ts
import pandas as pd

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--jplace', type=str)
parser.add_argument('--outfile', type=str)
args = parser.parse_args()

with open(args.jplace, 'r') as f:
    tree = json.load(f)

query_counter = {}

for p in tree['placements']:
    if p['n'][0] not in query_counter:
        query_counter[p['n'][0]] = p
    else:
        mse = p['p'][0][1]
        if mse < query_counter[p['n'][0]]['p'][0][1]:
            query_counter[p['n'][0]] = p

placements = [query_counter[q] for q in query_counter]

tree['placements'] = placements

with open(args.outfile, 'w') as f:
    json.dump(tree, f)