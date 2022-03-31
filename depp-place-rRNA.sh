#!/bin/bash

while getopts q:a:o:t:x:d: flag
do
    case "${flag}" in
	q) query_file=${OPTARG};;
	a) accessory_dir=${OPTARG};;
  o) out_dir=${OPTARG};;
  t) data_type=${OPTARG};;
  x) cores=${OPTARG};;
  d) debug=${OPTARG};;
    esac
done

# check if data type input is valid
valid_data="16s_full_length 16s_v4_100 16s_v4_150 16s_v3_v4 mixed"
#contains() {
#  [[ $1 =~ (^|[[:space:]])$2($|[[:space:]]) ]] && exit(0) || exit(1)
#}
function notcontains(){
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]];
  then
    result=1
  else
    result=0
  fi
  return $result
}
if `notcontains "${valid_data}" "${data_type}"`;
then
  echo "${data_type} is not supported currently"
  echo "please input a valid data_type (-t) in ${valid}"
  exit 1
fi

# check whether outdir exists, if not, create it
if [ ! -d "${out_dir}" ]; then
  echo "${out_dir} not exists, create ${out_dir}"
  mkdir -p ${out_dir}
fi

query_file=`realpath $query_file`
accessory_dir=`realpath $accessory_dir`
out_dir=`realpath $out_dir`

if [[ "${data_type}" == "mixed" ]];
then
  if [ -d "/scratch/$USER/job_$SLURM_JOB_ID" ];
  then
    # check whether ssds for tmp files exist
#    echo "use /scratch/$USER/job_$SLURM_JOB_ID as temporary directory"
    export TMPDIR=/scratch/$USER/job_$SLURM_JOB_ID
  else
#    echo "temporary file stores to $out_dir"
    export TMPDIR=${out_dir}
  fi
  tmpdir=`mktemp -d -t 'depp-tmp-XXXXXXXXXX'`
  # check if the directory is created
  if [[ ! "$tmpdir" || ! -d "$tmpdir" ]]; then
    echo "Could not create temp dir"
    exit 1
  fi
  mkdir ${tmpdir}/query_seq
  seq_sep.py --infile ${query_file} --outdir ${tmpdir}/query_seq

  for i in ${tmpdir}/query_seq/*.fa;
  do
    j="${i##*/}"
    dt="${j%.*}"
    depp-place-rRNA-one-type.sh -q ${i} -a ${accessory_dir} -o ${tmpdir} -t ${dt} -x ${cores}
    cp ${tmpdir}/${dt}_aligned.fa ${out_dir}/
#    mv ${tmpdir}/placement.jplace ${tmpdir}/${dt}.jplace
  done
  comb_json.py --indir ${tmpdir} --outfile ${tmpdir}/placement.jplace
  cp ${tmpdir}/placement.jplace ${out_dir}/
  gappa examine graft --jplace-path ${tmpdir}/placement.jplace  --out-dir ${out_dir}/ --allow-file-overwriting > /dev/null 2>&1
  function cleanup {
    rm -rf $tmpdir
    rm -rf $upptmpdir
    unset TMPDIR
  }
  trap cleanup EXIT
else
  if [ -d "/scratch/$USER/job_$SLURM_JOB_ID" ];
  then
    # check whether ssds for tmp files exist
#    echo "use /scratch/$USER/job_$SLURM_JOB_ID as temporary directory"
    export TMPDIR=/scratch/$USER/job_$SLURM_JOB_ID
  else
#    echo "temporary file stores to $out_dir"
    export TMPDIR=${out_dir}
  fi
  tmpdir=`mktemp -d -t 'depp-tmp-XXXXXXXXXX'`
  # check if the directory is created
  if [[ ! "$tmpdir" || ! -d "$tmpdir" ]]; then
    echo "Could not create temp dir"
    exit 1
  fi
  depp-place-rRNA-one-type.sh -q ${query_file} -a ${accessory_dir} -o ${tmpdir} -t $data_type -x ${cores}
  comb_json.py --indir ${tmpdir} --outfile ${tmpdir}/placement.jplace
  cp ${tmpdir}/placement.jplace ${out_dir}/
  cp ${tmpdir}/${data_type}_aligned.fa ${out_dir}/
  gappa examine graft --jplace-path ${tmpdir}/placement.jplace  --out-dir ${out_dir}/ --allow-file-overwriting > /dev/null 2>&1
fi
