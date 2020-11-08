# DEPP
## requirements
* Python 3
* [Newick Utilities](http://cegg.unige.ch/newick_utils) `conda install -c bioconda newick_utils`
* [gappa](https://github.com/lczech/gappa) `conda install -c bioconda gappa`

## Installation
`pip install depp`

## Usage
### Model training
`train_depp.py backbone_tree_file=$backbone_tree_file backbone_seq_file=$backbone_seq_file gpus=$gpus_id`
| arguments              | descriptions                                                                                                            |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **backbone_tree_file** | path to the backbone tree file (in **newick** format, **required**)                                                     |
| **backbone_seq_file**  | path to the backbone sequences file (in **fasta** format, **required**)                                                 |
| **model_dir**          | directory to save model's parameters for later used (default `model`)                                                   |
| **gpus**               | gpu ids (default `'[0]'`, **Don't** omit the quotes, i.e. use `gpus='[0]'` instead of `gpus=[0]`; if no gpu is available, use `gpus=0`; this version doesn't support multiple gpus temporarily, we will update it later)                                            |
| **embedding_size**     | embedding size (default: `128`)                                                                                         |
| **batch_size**         | batch size (default: `32`)                                                                                              |
| **resblock_num**       | number of residual blocks (default: `1`)                                                                                |

### Calculating distance matrix
`distance_depp.sh -q $query_seq_file -b $backbone_seq_file -m $model_path -t $backbone_tree_file -o $outdir`
| arguments             | descriptions                                                            |
|-----------------------|-------------------------------------------------------------------------|
| **-q**                | path to the query sequences file (in **fasta** format, **required**)    |
| **-b**                | path to the backbone sequences file (in **fasta** format, **required**) |
| **-m**                | path to the depp model's parameters file (**required**)                 |
| **-o**                | directory to store the output distance matrix (directory for output distance matrix, **required**) |
| **-t**                | path to the backbone tree file (in **newick** format, **required**).    |

Running the above command will give two distance matricies (`depp.csv` and `depp_correction.csv`), each as a tab delimited csv file with column and row headers. Rows represent query sequences and columns represent backbone sequences. `depp.csv` and `depp_correction.csv` represent distance_matrix before and after correction descripted in the paper.
