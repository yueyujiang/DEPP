#!/bin/bash

while getopts q:a:o:t:x:d:s:l: flag
do
    case "${flag}" in
	q) query_file=${OPTARG};;
	a) accessory_dir=${OPTARG};;
  o) out_dir=${OPTARG};;
  t) data_type=${OPTARG};;
  x) cores=${OPTARG};;
  d) debug=${OPTARG};;
  l) align=${OPTARG};;
#  s) script_dir=${OPTARG};;
    esac
done

align="${align:-noalign}"
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
#script_dir=`realpath $script_dir`
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
  if [ $align == "noalign" ];
  then
    mkdir ${tmpdir}/query_seq
    seq_sep.py --infile ${query_file} --outdir ${tmpdir}/query_seq

    # align sequences
    echo "aligning the sequences..."

    # split query sequences into multiple files (/tmp directory)
    for i in ${tmpdir}/query_seq/*.fa;
    do
      upptmpdir=`mktemp -d -t 'upp-tmp-XXXXXXXXXX'`
      # check if the directory is created
      if [[ ! "$tmpdir" || ! -d "$tmpdir" ]]; then
        echo "Could not create temp dir"
        exit 1
      fi
      j="${i##*/}"
      dt="${j%.*}"
      N=$(grep ">" ${i} | wc -l)
      awk -v size=4000000 -v pre=${upptmpdir}/seq -v pad="${#N}" '
         /^>/ { n++; if (n % size == 1) { close(fname); fname = sprintf("%s.%0" pad "d", pre, n) } }
            { print >> fname }
      ' ${i}

      mkdir ${upptmpdir}/aligned
      cnt=0
      for i in ${upptmpdir}/seq*;
      do
        mkdir ${upptmpdir}/tmp
        mkdir ${upptmpdir}/tmp_result
        grep ">" ${i} | sed "s/^>//g" | sort > ${upptmpdir}/tmp_result/query_ids.txt
        upp_c=${cores}
      #  awk "/^>/ {n++} n>8000 {exit} {print}" ${accessory_dir}/${data_type}_a.fasta > ${i}.backbone
        run_upp.py -s ${i} -a ${accessory_dir}/${dt}_ao.fasta -t ${accessory_dir}/${dt}.nwk -A 200 -d ${upptmpdir}/tmp_result -x $upp_c -p ${upptmpdir}/tmp 1>${tmpdir}/upp-out.log 2>>${out_dir}/${dt}_upp.log
  #      grep -w -A 1 -f ${upptmpdir}/xtmp_result/query_ids.txt ${upptmpdir}/tmp_result/output_alignment_masked.fasta --no-group-separator > ${upptmpdir}/aligned/${cnt}.fa
        seqkit grep -w 0 -f ${upptmpdir}/tmp_result/query_ids.txt ${upptmpdir}/tmp_result/output_alignment_masked.fasta -o ${upptmpdir}/aligned/${cnt}.fa
        cnt=$((cnt+1))
        rm -rf ${upptmpdir}/tmp
        rm -rf ${upptmpdir}/tmp_result
      done
      cat ${upptmpdir}/aligned/*.fa > ${tmpdir}/${dt}_aligned.fa
      rm -rf $upptmpdir
    done

    rm ${tmpdir}/query_seq/*
    if test -f "${tmpdir}/16s_full_length_aligned.fa";
    then
      cp ${tmpdir}/16s_full_length_aligned.fa ${tmpdir}/query_seq
    fi

    if test -f "${tmpdir}/16s_v4_aligned.fa";
    then
      cp ${tmpdir}/16s_v4_aligned.fa ${out_dir}/16s_v4_aligned_all.fa
      seq_sep.py --infile ${tmpdir}/16s_v4_aligned.fa --outdir ${tmpdir}/query_seq --aligned
    fi
  else
    mkdir -p ${tmpdir}/query_seq/
    cp $align/*aligned.fa ${tmpdir}/query_seq/
  fi

  for i in ${tmpdir}/query_seq/*.fa;
  do
    j="${i##*/}"
    dt="${j%_aligned.*}"
    depp-place-rRNA-one-type.sh -q ${i} -a ${accessory_dir} -o ${tmpdir} -t ${dt} -x ${cores}
    cp ${tmpdir}/${dt}_aligned.fa ${out_dir}/
#    cp ${tmpdir}/${dt}_upp.log ${out_dir}/${dt}_upp.log
#    mv ${tmpdir}/placement.jplace ${tmpdir}/${dt}.jplace
  done
  comb_json.py --indir ${tmpdir} --outfile ${tmpdir}/placement.jplace
  cp ${tmpdir}/placement.jplace ${out_dir}/
  cat ${tmpdir}/*entropy.txt > ${out_dir}/entropy.txt
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
  cat ${tmpdir}/*entropy.txt > ${out_dir}/entropy.txt
  cp ${tmpdir}/${data_type}_upp.log ${out_dir}
  gappa examine graft --jplace-path ${tmpdir}/placement.jplace  --out-dir ${out_dir}/ --allow-file-overwriting > /dev/null 2>&1
fi

cat ${out_dir}/*aligned.fa > ${out_dir}/aligned.fa
count_gapped_ratio.py --infile ${out_dir}/aligned.fa --outfile ${out_dir}/gap.txt
rm ${out_dir}/aligned.fa
filter_by_entropy_gap.sh 0.8 0.1 ${out_dir} ${query_file} ${accessory_dir}
filter_by_entropy_gap.sh 1 0.2 ${out_dir} ${query_file} ${accessory_dir}
filter_by_entropy_gap.sh 1.2 0.2 ${out_dir} ${query_file} ${accessory_dir}
filter_by_entropy_gap.sh 1.2 0.3 ${out_dir} ${query_file} ${accessory_dir}
mkdir ${out_dir}/all
cat ${query_file} ${accessory_dir}/seqs.fa  > ${out_dir}/all/seqs.fa
cp ${out_dir}/placement.jplace ${out_dir}/all/
cp ${accessory_dir}/mapfile.json ${out_dir}/all/
