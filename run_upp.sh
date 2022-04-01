#! /usr/bin/env bash

# $1 gene
# $2 accessory_dir
# $3 query_dir
# $4 cores
export UPPCORES=6 # 6 cores per upp job
export TMPDIR=/dev/shm
export TMP=$TMPDIR

if test -f ${2}/${1}_a.fasta; then
mkdir -p ${3}/${1}
grep ">" ${2}/${1}_a.fasta | sed "s/^>//g" | sort > ${3}/${1}/ref_ids.txt

mapfile -t < ${3}/${1}/ref_ids.txt
nw_prune -v <(nw_topology -bI ${2}/${1}.nwk) "${MAPFILE[@]}" > ${3}/${1}/backbone_ml.nwk
grep ">" ${3}/${1}.fa | sed "s/^>//g" | sort > ${3}/${1}/ref_ids.txt
if [ -z "${4+x}" ]
then
  run_upp.py -s ${3}/${1}.fa -a ${2}/${1}_a.fasta -t ${3}/${1}/backbone_ml.nwk -A 100 -d ${3}/${1} -d ${3}/${1} -p ${3}/${1}
else
  run_upp.py -s ${3}/${1}.fa -a ${2}/${1}_a.fasta -t ${3}/${1}/backbone_ml.nwk -A 100 -d ${3}/${1} -x $4 -d ${3}/${1} -p ${3}/${1}
fi
grep -w -A 1 -f ${3}/${1}/ref_ids.txt ${3}/${1}/output_alignment_masked.fasta --no-group-separator > ${3}/${1}_aligned.fa
rm -rf ${3}/${1}
fi
