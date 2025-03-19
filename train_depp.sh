#!/bin/bash

while getopts t:s:e:g:o:c:k:r: flag
do
    case "${flag}" in
	t) backbone_tree=${OPTARG};;
	s) backbone_seq=${OPTARG};;
  e) epochs=${OPTARG};;
  g) gpu=${OPTARG};;
  o) outdir=${OPTARG};;
  c) cluster=${OPTARG};;
  k) combine=${OPTARG};;
  r) replicate_seq=${OPTARG};;
#  s) script_dir=${OPTARG};;
    esac
done

gpu="${gpu:-0}"
outdir="${outdir:-model}"
cluster="${cluster:-False}"
combine="${combine:-False}"
replicate_seq="${replicate_seq:-False}"

#tmp_dir=${outdir}/tmpdir${RANDOM}
#mkdir $tmp_dir

mkdir -p ${outdir}

echo "combine" ${combine}
if [ "$combine" != "False" ];
then
  FastTree -nt ${backbone_seq} > ${outdir}/`basename ${backbone_seq}`.nwk
  TreeCluster.py -i ${outdir}/`basename ${backbone_seq}`.nwk -o ${outdir}/`basename ${backbone_seq}`.treecluster -m max -t 0.002
  if [ "$replicate_seq" == "False" ];
  then
    combine_similar_seq.py --tree ${backbone_tree} --seq ${backbone_seq} --outdir ${outdir} --cluster ${outdir}/`basename ${backbone_seq}`.treecluster
  else
    combine_similar_seq.py --tree ${backbone_tree} --seq ${backbone_seq} --outdir ${outdir} --cluster ${outdir}/`basename ${backbone_seq}`.treecluster  --replicate-seq
  fi
  backbone_tree="${outdir}/tree_fortrain.newick"
  backbone_seq="${outdir}/backbone_seq_lca.fa"
fi

echo "cluster" ${cluster}
if [ "$cluster" == "False" ];
then
  epochs="${epochs:-5000}"
  if [ $gpu == "nogpu" ];
    then
      train_depp.py backbone_seq_file=${backbone_seq} backbone_tree_file=${backbone_tree} model_dir=${outdir} epoch=${epochs} num_worker=0 gpus=0 replicate_seq=True
    else
      CUDA_VISIBLE_DEVICES=$gpu train_depp.py backbone_seq_file=${backbone_seq} backbone_tree_file=${backbone_tree} model_dir=${outdir} epoch=${epochs} num_worker=0 replicate_seq=True
  fi
else
  epochs="${epochs:-2000}"
  train_cluster_depp.sh -t ${backbone_tree} -s ${backbone_seq} -e ${epochs} -g ${gpu} -o ${outdir} -r ${replicate_seq}
fi

#rm -rf $tmp_dir