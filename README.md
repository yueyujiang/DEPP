# DEPP
## requirements
* Python 3
* [Newick Utilities](http://cegg.unige.ch/newick_utils) `conda install -c bioconda newick_utils`
* [gappa](https://github.com/lczech/gappa) `conda install -c bioconda gappa`

## Installation
`pip install depp`

## Usage
### Sequences analysis using WoL data
We provide the pretrained model for WoL marker genes and ASV data. Users can place the query sequences onto the WoL species tree directly using DEPP.  
#### Preparation
* Install UPP following the instructions [here](https://github.com/smirarab/sepp/blob/master/README.UPP.md), make sure that run_upp.py is executable.  
* Sequences can be either unaligned ASV (16S) or unaligned MAG data or both.
* Marker genes    
  - Identify the marker genes using the protocols from [WoL project](https://biocore.github.io/wol/protocols/).  
  - Rename each sequence file using the the format: <marker gene's id>.fa, e.g. p0000.fa, p0001.fa...  
* ASV  
  - Models of five types of 16S data is pretrained: full-length (~1600bp), V4 region (~250bp), V3+V4 region (~400bp), V4 100 (~100bp), V4 150 (~150bp). (If your ASV data is in the above five types, you can analyze your data directly. Otherwise, please align your sequences and then train your own model using the `train_depp.py` command)  
  - Rename your ASV data using the following rules:  
    - full-length 16S: 16s_full_length.fa  
    - V3+V4 region: 16s_v3_v4.fa  
    - V4 region: 16s_v4.fa  
    - V4 100bp: 16s_v4_100.fa  
    - V4 100bp: 16s_v4_150.fa  
* Put all your query sequences files into one empty directory.  
* Download the models and auxiliary data (accessory.tar.gz) from [here](https://tera-trees.com/data/depp/latest/) and unzip it.  

#### Running
`wol_placement.sh -q directory/to/query/sequences -o directory/for/output -a directory/to/auxiliary/data/accessory`  
This command will give you a output directory named `depp_results`. items inside the directory include:  
* `summary` directory:  
  - placement tree in jplace and newick format for each sequences file.  
  - placement tree in jplace and newick format that include all the queries from all the files provided  
* each sequences file will have a directory which includes the distacne matrix from queries to backbone species.  

### Model training
`train_depp.py backbone_tree_file=backbone/tree/file backbone_seq_file=backbone/seq/file gpus=$gpus_id`
| arguments              | descriptions                                                                                                            |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **backbone_tree_file** | path to the backbone tree file (in **newick** format, **required**)                                                     |
| **backbone_seq_file**  | path to the backbone sequences file (in **fasta** format, **required**)                                                 |
| **model_dir**          | directory to save model's parameters for later used (default `model`)                                                   |
| **gpus**               | gpu ids (default `'[0]'`, **Don't** omit the quotes, i.e. use `gpus='[0]'` instead of `gpus=[0]`; **if no gpu is available**, use `gpus=0`; this version doesn't support multiple gpus temporarily, we will update it later)                                            |
| **embedding_size**     | embedding size <a href="https://www.codecogs.com/eqnedit.php?latex=2^{\frac{1}{2}\lfloor{\log_2&space;(100&space;n)}\rfloor}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{\frac{1}{2}\lfloor{\log_2&space;(100&space;n)}\rfloor}" title="2^{\frac{1}{2}\lfloor{\log_2 (100 n)}\rfloor}" /></a>, n is the number of leaves in the backbone tree).                                                                                         |
| **batch_size**         | batch size (default: `32`)                                                                                              |
| **resblock_num**       | number of residual blocks (default: `1`)                                                                                |

### Calculating distance matrix
`depp_distance.py backbone_seq_file=backbone/seq/file query_seq_file=query/seq/file model_path=model/path`
| arguments              | descriptions                                                                                                            |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **backbone_seq_file**  | path to the backbone sequences file (in **fasta** format, **required**)                                                 |
| **query_seq_file**  | path to the query sequences file (in **fasta** format, **required**)                                                 |
| **model_path**               | path to the trained model (**required**)                                     |

<!-- 
`distance_depp.sh -q query/seq/file -b $backbone/seq/file -m model/path -t backbone/tree/file -o $outdir`
| arguments             | descriptions                                                            |
|-----------------------|-------------------------------------------------------------------------|
| **-q**                | path to the query sequences file (in **fasta** format, **required**)    |
| **-b**                | path to the backbone sequences file (in **fasta** format, **required**) |
| **-m**                | path to the depp model's parameters file (**required**)                 |
| **-o**                | directory to store the output distance matrix (directory for output distance matrix, **required**) |
| **-t**                | path to the backbone tree file (in **newick** format, **required**).    | -->

Running the above command will generate a distance matricies (`depp.csv`), as a tab delimited csv file with column and row headers. Rows represent query sequences and columns represent backbone sequences.

Any questions? Please contact <y5jiang@eng.ucsd.edu>.
