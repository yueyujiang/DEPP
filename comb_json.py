#!/usr/bin/env python3

import argparse
import json
import os

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--indir', type=str)
parser.add_argument('--outfile', type=str)
args = parser.parse_args()

cnt = 0
for file in os.listdir(args.indir):
    if not file.endswith('jplace'):
        continue
    with open(f'{args.indir}/{file}', 'r') as f:
        if cnt == 0:
            tree = json.load(f)
        else:
            tree['placements'] += json.load(f)['placements']
        cnt += 1

with open(args.outfile, 'w') as f:
    json.dump(tree, f)