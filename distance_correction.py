import os
import Model_pl
import utils
import pandas as pd
from omegaconf import OmegaConf
import treeswift
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def main():
    args_base = OmegaConf.load('./config/default_config.yaml')

    args_cli = OmegaConf.from_cli()

    args = OmegaConf.merge(args_base, args_cli)
    original_distance = pd.read_csv(os.path.join(args.outdir, "depp.csv"), sep='\t')
    s = list(original_distance.keys())[1:]
    tree = treeswift.read_tree(args.backbone_tree, 'newick')
    true_max = 2 * tree.height()
    data = {}
    for i in range(len(original_distance)):
        line = list(original_distance.iloc[i])
        seq_name = line[0]
        with open(f"{args.outdir}/depp_tmp/{seq_name}_leaves.txt", "r") as f:
            method = set(f.read().split("\n"))
            method.remove('')
        query_median = np.median(original_distance[np.array(method)].iloc[i])
        ratio = true_max / (query_median + 1e-7)
        b = original_distance.iloc[0].values[1:] * ratio
        seq_dict = dict(zip(s, b))
        data[seq_name] = seq_dict
    data = pd.DataFrame.from_dict(data, orient='index', columns=s)    
    data.to_csv(os.path.join(args.outdir, f'depp_correction.csv'), sep='\t')
    
if __name__ == '__main__':
    main()

