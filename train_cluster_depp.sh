#!/bin/bash

# $1 backbone tree
# $2 backbone sequences
# $3 outdir
echo preparing the data ...
echo clustering the tree ...
while getopts t:s:e:g:o:r: flag
do
    case "${flag}" in
	t) backbone_tree=${OPTARG};;
	s) backbone_seq=${OPTARG};;
  e) epochs=${OPTARG};;
  g) gpu=${OPTARG};;
  o) outdir=${OPTARG};;
  r) replicate_seq=${OPTARG};;
#  s) script_dir=${OPTARG};;
    esac
done

epochs="${epochs:-2000}"
gpu="${gpu:-0}"
outdir="${outdir:-model}"
replicate_seq="${replicate_seq}:-False"

tmp_dir=${outdir}/tmpdir${RANDOM}

rm -rf $tmp_dir
mkdir -p $tmp_dir

if [ "${replicate_seq}" == "False" ];
then
  grep ">" ${backbone_seq} | sed "s/>//g" > ${tmp_dir}/seq.label
else
  grep ">" ${backbone_seq} | sed "s/>//g" | sed "s/_[^ \t]*//g" | sort | uniq > ${tmp_dir}/seq.label
fi
nw_prune -vf $backbone_tree ${tmp_dir}/seq.label > $tmp_dir/backbone.nwk
backbone_tree=$tmp_dir/backbone.nwk
set_bl_one.py --infile $backbone_tree --outfile $tmp_dir/backbone_blone.nwk
TreeCluster.py -i ${tmp_dir}/backbone_blone.nwk -o ${tmp_dir}/treecluster.txt -m sum_branch -t 3000
mkdir ${tmp_dir}/test_labels
while IFS=$'\t' read -r seq_name cluster_number; do
    # Skip the header line
    if [ "$seq_name" != "SequenceName" ]; then
        # Append the sequence name to the corresponding cluster file, creating a new line for each sequence name
        echo "$seq_name" >> "${tmp_dir}/test_labels/$((cluster_number-1)).txt"
    fi
done < "${tmp_dir}/treecluster.txt"

mkdir ${tmp_dir}/test_trees
for i in ${tmp_dir}/test_labels/*;
do
  name=`basename $i`
  id=${name%.*}
  nw_prune -vf $backbone_tree $i > ${tmp_dir}/test_trees/${id}.nwk
done

echo spliting the tree into smaller clusters ...
mkdir ${tmp_dir}/test_trees_subtree ${tmp_dir}/test_labels_subtree
echo "{" > ${tmp_dir}/subtree_corr.txt
echo "" >> ${tmp_dir}/subtree_corr.txt
for subtree in ${tmp_dir}/test_trees/*;
do
        i=`basename ${subtree}`
        i=${i%.*}
        #echo "subtree" ${subtree}
        set_bl_one.py --infile ${subtree} --outfile ${tmp_dir}/test_trees_subtree/${i}_blone.nwk
        TreeCluster.py -i ${tmp_dir}/test_trees_subtree/${i}_blone.nwk -o ${tmp_dir}/test_trees_subtree/treecluster_subtree${i}.txt -m sum_branch -t 60
        mkdir ${tmp_dir}/test_labels_subtree_tmp

        while IFS=$'\t' read -r seq_name cluster_number; do
            # Skip the header line
            if [ "$seq_name" != "SequenceName" ]; then
                # Append the sequence name to the corresponding cluster file, creating a new line for each sequence name
                echo "$seq_name" >> "${tmp_dir}/test_labels_subtree_tmp/$((cluster_number-1)).txt"
            fi
        done < "${tmp_dir}/test_trees_subtree/treecluster_subtree${i}.txt"

        cur_idx=`ls -1 ${tmp_dir}/test_labels_subtree | wc -l`
        echo "\"${i}\":[" >> ${tmp_dir}/subtree_corr.txt
        for label in ${tmp_dir}/test_labels_subtree_tmp/*;
        do
                j=`basename ${label}`
                j=${j%.*}
                #echo "j" ${j} ${cur_idx}
                echo "$((j+cur_idx))," >> ${tmp_dir}/subtree_corr.txt
                mv ${label} ${tmp_dir}/test_labels_subtree/$((j+cur_idx)).txt
        done
        echo "]," >> ${tmp_dir}/subtree_corr.txt
        rm -rf ${tmp_dir}/test_labels_subtree_tmp ${tmp_dir}/test_trees_subtree/${i}_blone.nwk ${tmp_dir}/test_trees_subtree/treecluster_subtree${i}.txt
done

echo "}" >> ${tmp_dir}/subtree_corr.txt
tr "\n" " " < ${tmp_dir}/subtree_corr.txt | sed "s/, ]/]/g" | sed "s/, }/}/g" > ${tmp_dir}/subtree_corr.txt.tmp
mv ${tmp_dir}/subtree_corr.txt.tmp ${tmp_dir}/subtree_corr.txt
mkdir ${tmp_dir}/test_seqs_subtree

s=""
for label in ${tmp_dir}/test_labels_subtree/*;
do
        s="${s} ${label}"
done

grep_seq_group.py --infile $backbone_seq --outdir ${tmp_dir}/test_seqs_subtree --name-list ${s}

echo adding representatives ...
mkdir ${tmp_dir}/add_repr_label
#for label in ${1}/test_labels/*;
for label in ${tmp_dir}/test_labels/*;
do
        s=""
        i=`basename ${label}`
        i=${i%.*}
        for label2 in ${tmp_dir}/test_labels/*;
        do
                j=`basename ${label2}`
                j=${j%.*}
                if [ "${j}" -eq "${i}" ];
                then
                        continue
                fi
                if test -f "${tmp_dir}/current_tree${j}_${i}.csv";
                then
                        s="${s} ${tmp_dir}/current_tree${j}_${i}.csv"
                        continue
                fi
                if test -f "${tmp_dir}/current_tree${i}_${j}.csv";
                then
                        s="${s} ${tmp_dir}/current_tree${i}_${j}.csv"
                        continue
                fi
                cat ${label} > ${tmp_dir}/current_label.txt
                echo "" >> ${tmp_dir}/current_label.txt
                cat ${label2} >> ${tmp_dir}/current_label.txt
                nw_prune -v $backbone_tree `cat ${tmp_dir}/current_label.txt` > ${tmp_dir}/current_tree.nwk
                get_tree_dist.py --treefile ${tmp_dir}/current_tree.nwk --outfile ${tmp_dir}/current_tree${i}_${j}.csv
                s="${s} ${tmp_dir}/current_tree${i}_${j}.csv"
        done
        #echo distlist ${s}
        select_species_by_distance.py --distlist ${s} --outfile ${tmp_dir}/add_repr_label/${i}.txt --cluster ${label}
        #rm ${tmp_dir}/current_tree.nwk ${tmp_dir}/current_label.txt
done

mkdir ${tmp_dir}/test_trees_add_repr
mkdir ${tmp_dir}/test_seqs_add_repr
mkdir ${tmp_dir}/test_labels_add_repr
s=""
for label in ${tmp_dir}/add_repr_label/*
do
  i=`basename ${label}`
  i=${i%.*}
  cat ${label} ${tmp_dir}/test_labels/${i}.txt > ${tmp_dir}/test_labels_add_repr/${i}.txt
  nw_prune -v $backbone_tree `cat ${tmp_dir}/test_labels_add_repr/${i}.txt` > ${tmp_dir}/test_trees_add_repr/${i}.nwk
  s="${s} ${tmp_dir}/test_labels_add_repr/${i}.txt"
done

grep_seq_group.py --infile $backbone_seq --outdir ${tmp_dir}/test_seqs_add_repr --name-list ${s}


mv ${tmp_dir}/test_seqs_add_repr ${outdir}/test_seqs_add_repr
mv ${tmp_dir}/test_trees_add_repr ${outdir}/test_trees_add_repr
mv ${tmp_dir}/test_seqs_subtree ${outdir}/test_seqs_subtree
mv ${tmp_dir}/subtree_corr.txt ${outdir}/subtree_corr.txt
rm -rf ${tmp_dir}

echo finish data preparing!
echo start the training

num_cluster=`ls -1 ${outdir}/test_trees_add_repr/*nwk | wc -l`
if [ $gpu == "nogpu" ];
  then
    train_depp.py seqdir=${outdir}/test_seqs_add_repr treedir=${outdir}/test_trees_add_repr model_dir=${outdir} classifier_epoch=100 epoch=${epochs} cluster_num=${num_cluster} num_worker=0 classifier_seqdir=${outdir}/test_seqs_subtree cluster_corr=${outdir}/subtree_corr.txt gpus=0 replicate_seq=True
  else
    CUDA_VISIBLE_DEVICES=$gpu train_depp.py seqdir=${outdir}/test_seqs_add_repr treedir=${outdir}/test_trees_add_repr model_dir=${outdir} classifier_epoch=100 epoch=${epochs} cluster_num=${num_cluster} num_worker=0 classifier_seqdir=${outdir}/test_seqs_subtree cluster_corr=${outdir}/subtree_corr.txt replicate_seq=True
fi