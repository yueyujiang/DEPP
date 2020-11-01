# DEPP
## requirements
* Python 3
* PyTorch
* Numpy
* Pandas
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [Treeswift](https://github.com/niemasd/TreeSwift)
* [biopython](https://biopython.org/wiki/Download)
* [OmegaConf](https://github.com/omry/omegaconf)

## Usage
### Model training
`python train_pl.py backbone_tree_file=$backbone_tree_file backbone_seq_file=$backbone_seq_file gpus=$gpus_id`
| arguments              | descriptions                                                                                                            |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **backbone_tree_file** | path to the backbone tree file (in **newick** format, **required**)                                                     |
| **backbone_seq_file**  | path to the backbone sequences file (in **fasta** format, **required**)                                                 |
| **model_dir**          | directory to save model's parameters for later used (default *model*)                                                   |
| **gpus**               | gpu ids (default `'[0]'`, if no gpu is available, use `gpus=None`, this version is not compatible with multiple gpus)   |
| **embedding_size**     | embedding size (default: `128`)                                                                                         |
| **batch_size**         | batch size (default: `32`)                                                                                              |
| **resblock_num**       | number of residual blocks (default: `1`)                                                                                |

### Calculating distance matrix
`python depp_distance.py query_seq_file=$query_seq_file model_path=$model_parameters_file backbone_seq_file=$backbone_seq_file`
| arguments             | descriptions                                                            |
|-----------------------|-------------------------------------------------------------------------|
|   **query_seq_file**  | path to the query sequences file (in **fasta** format, **required**)    |
| **backbone_seq_file** | path to the backbone sequences file (in **fasta** format, **required**) |
| **model_path**        | path to the depp model's parameters file (**required**)                 |
| **outdir**            | directory to store the output distance (default `depp_distance`)        |

Running depp_distance.py will give two distance matricies (`depp.csv` and `depp_correction.csv`), each as a tab delimited csv file with column and row headers. Rows represent query sequences and columns represent backbone sequences. `depp.csv` and `depp_correction.csv` represent distance_matrix before and after correction descripted in the paper.
