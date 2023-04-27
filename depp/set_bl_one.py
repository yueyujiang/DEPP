#!/usr/bin/env python3

import argparse
import treeswift as ts

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--infile', type=str)
parser.add_argument('--outfile', type=str)
args = parser.parse_args()

tree = ts.read_tree_newick(args.infile)
for n in tree.traverse_preorder():
    n.edge_length = 1 

tree.write_tree_newick(args.outfile)
