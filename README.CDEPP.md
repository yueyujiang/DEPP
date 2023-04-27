# This is the implementation of C-DEPP 
https://www.biorxiv.org/content/10.1101/2023.03.27.534201v2

## Install      
`pip install depp==0.3.1`

## Usage
### Training        
example           
`train_cluster_depp.sh -t test/cluster_depp/backbone.nwk -s test/cluster_depp/backbone.fa -g 2 -o outdir`        
* `-t` tree file        
* `-s` sequences file           
* `-g` gpu id. If no gpu is used, use `-g nogpu`, default 0                 
* `-o` output directory
* `-e` number of training epochs, default 2000.

### Testing
example          
`depp_distance.py seqdir=outdir/test_seqs_add_repr query_seq_file=test/cluster_depp/query.fa model_path=outdir/cluster-depp.pth outdir=outdir use_multi_class=True prob_thr=200`        
This will produce distance files in directory `outdir` named `depp*csv`
