#!/usr/bin/python
import os
import pandas as pd
import treeswift
import numpy as np
import default_config
from omegaconf import OmegaConf

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def main():
    args_base = OmegaConf.create(default_config.default_config)

    args_cli = OmegaConf.from_cli()

    args = OmegaConf.merge(args_base, args_cli)
    original_distance = pd.read_csv(os.path.join(args.outdir, "depp.csv"), sep='\t')
    a_for_seq_name = pd.read_csv(os.path.join(args.outdir, "depp.csv"), sep='\t', dtype=str)
    s = list(original_distance.keys())[1:]
    tree = treeswift.read_tree(args.backbone_tree, 'newick')
    true_max = tree.diameter()
    # print(true_max)
    data = {}
    s_set = set(s)
    for i in range(len(original_distance)):
        line = list(a_for_seq_name.iloc[i])
        seq_name = line[0]
        with open(f"{args.outdir}/depp_tmp/{seq_name}_leaves.txt", "r") as f:
            method = set(f.read().split("\n"))
            method.remove('')
            method = method.intersection(s_set)
        if method:
            query_median = np.median(original_distance[np.array(method)].iloc[i])
            ratio = true_max / (query_median + 1e-7)
            # print(ratio)
            b = original_distance.iloc[i].values[1:] * ratio
        else:
            b = original_distance.iloc[i].values[1:]
        seq_dict = dict(zip(s, b))
        data[seq_name] = seq_dict
    data = pd.DataFrame.from_dict(data, orient='index', columns=s)    
    data.to_csv(os.path.join(args.outdir, f'depp_correction.csv'), sep='\t')
    
if __name__ == '__main__':
    main()

