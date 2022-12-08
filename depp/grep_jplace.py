#!/usr/bin/env python3

import json
import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--infile', type=str)
parser.add_argument('--outfile', type=str)
parser.add_argument('--names', type=str)
args = parser.parse_args()

with open(args.infile, 'r') as f:
    tree = json.load(f)

with open(args.names, 'r') as f:
    names = set(f.read().split('\n'))

placements = [i for i in tree['placements'] if i['n'][0] in names]

tree['placements'] = placements

with open(args.outfile, 'w') as f:
    json.dump(tree, f, sort_keys=True, indent=4)
