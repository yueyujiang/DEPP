#!/bin/bash

while getopts q:a:o:t:x:d:s: flag
do
    case "${flag}" in
	q) query_file=${OPTARG};;
	a) accessory_dir=${OPTARG};;
  o) out_dir=${OPTARG};;
  t) data_type=${OPTARG};;
  x) cores=${OPTARG};;
  d) debug=${OPTARG};;
#  s) script_dir=${OPTARG};;
    esac
done

if [ -d "/scratch/$USER/job_$SLURM_JOB_ID" ];
then
  # check whether ssds for tmp files exist
#  echo "use /scratch/$USER/job_$SLURM_JOB_ID as temporary directory"
  export TMPDIR=/scratch/$USER/job_$SLURM_JOB_ID
else
#  echo "temporary file stores to $out_dir"
  export TMPDIR=${out_dir}
fi

tmpdir=`mktemp -d -t 'depp-tmp-XXXXXXXXXX'`
# check if the directory is created
if [[ ! "$tmpdir" || ! -d "$tmpdir" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

upptmpdir=`mktemp -d -t 'upp-tmp-XXXXXXXXXX'`
# check if the directory is created
if [[ ! "$tmpdir" || ! -d "$tmpdir" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

export query_file=`realpath $query_file`
export accessory_dir=`realpath $accessory_dir`
export out_dir=`realpath $out_dir`
export c="${cores:-4}"
#export script_dir=`realpath $script_dir`
export data_type
export tmpdir
export upptmpdir

# check whether output directory exist
if [ ! -d "$out_dir" ]; then
  echo "$out_dir not exist"
  exit 1
fi

echo "start ${data_type} data placement"

## align sequences
#echo "aligning the sequences..."
#
## split query sequences into multiple files (/tmp directory)
#N=$(grep ">" ${query_file} | wc -l)
#awk -v size=4000000 -v pre=${upptmpdir}/seq -v pad="${#N}" '
#   /^>/ { n++; if (n % size == 1) { close(fname); fname = sprintf("%s.%0" pad "d", pre, n) } }
#      { print >> fname }
#' ${query_file}
#
#mkdir ${upptmpdir}/aligned
#cnt=0
#for i in ${upptmpdir}/seq*;
#do
#  mkdir ${upptmpdir}/tmp
#  mkdir ${upptmpdir}/tmp_result
#  grep ">" ${i} | sed "s/^>//g" | sort > ${upptmpdir}/tmp_result/query_ids.txt
#  upp_c=$c
##  awk "/^>/ {n++} n>8000 {exit} {print}" ${accessory_dir}/${data_type}_a.fasta > ${i}.backbone
#  run_upp.py -s ${i} -a ${accessory_dir}/${data_type}_ao.fasta -t ${accessory_dir}/${data_type}.nwk -A 200 -d ${upptmpdir}/tmp_result -x $upp_c -p ${upptmpdir}/tmp 1>${tmpdir}/upp-out.log 2>>${tmpdir}/upp-err.log
#  grep -w -A 1 -f ${upptmpdir}/tmp_result/query_ids.txt ${upptmpdir}/tmp_result/output_alignment_masked.fasta --no-group-separator > ${upptmpdir}/aligned/${cnt}.fa
#  cnt=$((cnt+1))
#  rm -rf ${upptmpdir}/tmp
#  rm -rf ${upptmpdir}/tmp_result
#done
#cat ${upptmpdir}/aligned/*.fa > ${tmpdir}/${data_type}_aligned.fa
#rm -rf $upptmpdir

cp ${query_file} ${tmpdir}/

# split large file into multiple small ones
N=$(grep ">" ${query_file} | wc -l)
mkdir ${tmpdir}/split_seq
awk -v size=20000 -v pre=${tmpdir}/split_seq/seq -v pad="${#N}" '
   /^>/ { n++; if (n % size == 1) { close(fname); fname = sprintf("%s.%0" pad "d", pre, n) } }
      { print >> fname }
' ${tmpdir}/${data_type}_aligned.fa

# calculate distance

function dist_place(){
  local seq_id=$1
  local c=$2
  local accessory_dir=$3
  local data_type=$4
  local tmpdir=$5
  local idx=$6
  taskset -c $(((idx*2)%c))-$(((idx*2)%c+1)) depp_distance.py query_seq_file=$1 outdir=../dist/$1 model_path=${accessory_dir}/${data_type}.ckpt replicate_seq=True backbone_emb=${accessory_dir}/${data_type}_emb.pt backbone_id=${accessory_dir}/${data_type}_id.pt
  jplace_suffix="${seq_id##*.}"
  apples_core=`python -c "print(min($c,2))"`
  mkdir ${tmpdir}/${data_type}_${jplace_suffix}
  for depp_dist_file in ../dist/$1/depp*csv;
  do
    depp_dist_idx=`basename $depp_dist_file`
    depp_dist_idx=${depp_dist_idx%%.*}
    taskset -c $(((idx*2)%c))-$(((idx*2)%c+1)) run_apples.py -d $depp_dist_file -t ${accessory_dir}/wol.nwk -o ${tmpdir}/${data_type}_${jplace_suffix}/${data_type}_${jplace_suffix}_${depp_dist_idx}.jplace -f 0 -b 5 -T $apples_core
  done
  cp ../dist/$1/entropy.txt ${tmpdir}/${data_type}_${jplace_suffix}_entropy.txt
  taskset -c $(((idx*2)%c))-$(((idx*2)%c+1)) comb_json.py --indir ${tmpdir}/${data_type}_${jplace_suffix} --outfile ${tmpdir}/${data_type}_${jplace_suffix}.jplace
  taskset -c $(((idx*2)%c))-$(((idx*2)%c+1)) rm -rf ../dist/$1
  taskset -c $(((idx*2)%c))-$(((idx*2)%c+1)) rm -rf ${tmpdir}/${data_type}_${jplace_suffix}
#  taskset -c $(((idx*4)%c))-$(((idx*4)%c+3)) rm ../dist/$1/*.pt
}
export -f dist_place

echo "calculating distance matrix..."
pushd ${tmpdir}/split_seq/ > /dev/null 2>&1
depp_p=`python -c "print(max($c//2,1))"`
#ls -1 seq* | xargs -n1 -P$depp_p -I% bash -c 'dist_place "%" "${c}" "${accessory_dir}" "${data_type}" "${tmpdir}"'
n_seq=`ls -1 seq*|wc -l`
num_jobs="\j"
j=0
for ((j=0; j<$n_seq; j++));
do
        while (( ${num_jobs@P} >= $depp_p ));
        do
                wait -n
        done
        ( s_tmp=`ls -1 seq*| sed "$((j+1))q;d"` &&
	dist_place "${s_tmp}" "${c}" "${accessory_dir}" "${data_type}" "${tmpdir}" "${j}" ) &
done
wait
popd > /dev/null 2>&1


#cp ${tmpdir}/*.jplace ${out_dir}/
comb_json.py --indir ${tmpdir} --outfile ${out_dir}/placement_${data_type}.jplace
#cp ${tmpdir}/placement.newick ${out_dir}/
cp ${tmpdir}/${data_type}_aligned.fa ${out_dir}/
#cp ${tmpdir}/upp-err.log ${out_dir}/${data_type}_upp.log
#cp ${tmpdir}/dist/depp.csv ${out_dir}
cat ${tmpdir}/*_entropy.txt > ${out_dir}/${data_type}_entropy.txt

echo "finished!"

function cleanup {
    rm -rf $tmpdir
    rm -rf $upptmpdir
    unset TMPDIR
    unset query_file
    unset out_dir
    unset c
    unset data_type
    unset tmpdir
    unset upptmpdir
}
trap cleanup EXIT
