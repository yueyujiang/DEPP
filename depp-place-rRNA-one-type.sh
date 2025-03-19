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
#  export TMPDIR=${out_dir}
  TMPDIR=$(mktemp -d)/$(basename ${out_dir})
  mkdir -p ${TMPDIR}
  export TMPDIR=${TMPDIR}
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

echo "start query placement..."

cp ${query_file} ${tmpdir}/

# split large file into multiple small ones
N=$(grep ">" ${query_file} | wc -l)
mkdir ${tmpdir}/split_seq
awk -v size=2000 -v pre=${tmpdir}/split_seq/seq -v pad="${#N}" '
   /^>/ { n++; if (n % size == 1) { close(fname); fname = sprintf("%s.%0" pad "d", pre, n) } }
      { print >> fname }
' ${tmpdir}/${data_type}_aligned.fa

function dist_place(){
  local seq_id=$1
  local c=$2
  local accessory_dir=$3
  local data_type=$4
  local tmpdir=$5
  depp_distance.py query_seq_file=$1 outdir=../dist/$1 model_path=${accessory_dir}/${data_type}.ckpt replicate_seq=True backbone_emb=${accessory_dir}/${data_type}_emb.pt backbone_id=${accessory_dir}/${data_type}_id.pt use_multi_class=True prob_thr=200
  jplace_suffix="${seq_id##*.}"
  apples_core=`python -c "print(min($c,4))"`
  mkdir ${tmpdir}/${data_type}_${jplace_suffix}
  for depp_dist_file in ../dist/$1/depp*csv;
  do
    depp_dist_idx=`basename $depp_dist_file`
    depp_dist_idx=${depp_dist_idx%%.*}
    run_apples.py -d $depp_dist_file -t ${accessory_dir}/wol.nwk -o ${tmpdir}/${data_type}_${jplace_suffix}/${data_type}_${jplace_suffix}_${depp_dist_idx}.jplace -f 0 -b 5 -T $apples_core
  done
  cp ../dist/$1/entropy.txt ${tmpdir}/${data_type}_${jplace_suffix}_entropy.txt
  comb_json.py --indir ${tmpdir}/${data_type}_${jplace_suffix} --outfile ${tmpdir}/${data_type}_${jplace_suffix}.jplace
  rm -rf ../dist/$1
  rm -rf ${tmpdir}/${data_type}_${jplace_suffix}
}
export -f dist_place

echo "calculating distance matrix..."
pushd ${tmpdir}/split_seq/ > /dev/null 2>&1

my_dir="./"
k=`python -c "print(min($c,4))"`  # Set the number of cores you want to use

# Calculate the number of files and jobs per iteration
files=("$my_dir"/seq*)
num_files=${#files[@]}
jobs_per_iteration=$(($c / $k))
#echo jobs_per_iteration $jobs_per_iteration

# Run the jobs in parallel
for ((i = 0; i < $num_files; i += $jobs_per_iteration)); do
    # Get a subset of files for the current iteration
    subset=("${files[@]:$i:$jobs_per_iteration}")

    # Run my_function in parallel for the subset of files
    echo "${subset[@]}" | xargs -n 1 -P $jobs_per_iteration -I {} bash -c 'dist_place  "$@"' _ {} "${c}" "${accessory_dir}" "${data_type}" "${tmpdir}"
done

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
