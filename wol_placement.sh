while getopts q:a:l:o:x:d:f: flag
do
    case "${flag}" in
	q) query_dir=${OPTARG};;
	a) accessory_dir=${OPTARG};;
  l) aligned=${OPTARG};;
  o) out_dir=${OPTARG};;
  x) cores=${OPTARG};;
  d) debug=${OPTARG};;
  f) filter=${OPTARG};;
    esac
done
out_dir=${out_dir}/depp_results
count1=`ls -1 ${query_dir}/p* 2>/dev/null | wc -l`
count2=`ls -1 ${query_dir}/16* 2>/dev/null | wc -l`

if [ ! -d "${query_dir}" ];
then
	echo $query_dir not exist
	exit 1
fi

if [ ! -d ${accessory_dir} ];
then
	echo ${accessory_dir} not exist
	exit 1
fi

if [ -z "${aligned+x}" ]
then
  echo "aligning sequences..."
  for f in ${query_dir}/*.fa; do
    # $1 gene
    # $2 accessory_dir
    # $3 query_dir
    # $4 out_dir
    f=$(basename -- "$f")
    f="${f%.*}"
#    echo $f
    if [ -z "${debug+x}" ]
    then
      run_upp.sh ${f} ${accessory_dir} ${query_dir} ${cores} > /dev/null 2>&1
    else
      run_upp.sh ${f} ${accessory_dir} ${query_dir} ${cores}
    fi
  done
  wait
  echo "finish alignment!"
fi
echo "calculating distance matrices..."
if [ -z "${aligned+x}" ]
then
  if [ $count1 -ne 0 ]
  then
  for f in ${query_dir}/p*aligned.fa; do
    f=$(basename -- "$f")
    f="${f%_*}"
    if [ -z "${debug+x}" ]
    then
      depp_distance.py query_seq_file=${query_dir}/${f}_aligned.fa backbone_seq_file=${accessory_dir}/${f}_a.fasta model_path=${accessory_dir}/${f}.ckpt outdir=${out_dir}/${f} > /dev/null 2>&1
    else
      depp_distance.py query_seq_file=${query_dir}/${f}_aligned.fa backbone_seq_file=${accessory_dir}/${f}_a.fasta model_path=${accessory_dir}/${f}.ckpt outdir=${out_dir}/${f}
    fi
  done
  wait
  fi
  if [ $count2 -ne 0 ]
  then
  for f in ${query_dir}/16*aligned.fa; do
    f=$(basename -- "$f")
    f="${f%_*}"
    if [ -z "${debug+x}" ]
    then
      depp_distance.py query_seq_file=${query_dir}/${f}_aligned.fa backbone_seq_file=${accessory_dir}/${f}_a.fasta model_path=${accessory_dir}/${f}.ckpt outdir=${out_dir}/${f} replicate=True > /dev/null 2>&1
    else
      depp_distance.py query_seq_file=${query_dir}/${f}_aligned.fa backbone_seq_file=${accessory_dir}/${f}_a.fasta model_path=${accessory_dir}/${f}.ckpt outdir=${out_dir}/${f} replicate=True
    fi
  done
  wait
  fi
else
  if [ $count1 -ne 0 ]
  then
  for f in ${query_dir}/p*.fa; do
    f=$(basename -- "$f")
    f="${f%.*}"
    if [ -z "${debug+x}" ]
    then
      depp_distance.py query_seq_file=${query_dir}/${f}.fa backbone_seq_file=${accessory_dir}/${f}_a.fasta model_path=${accessory_dir}/${f}.ckpt outdir=${out_dir}/${f} > /dev/null 2>&1
    else
      depp_distance.py query_seq_file=${query_dir}/${f}.fa backbone_seq_file=${accessory_dir}/${f}_a.fasta model_path=${accessory_dir}/${f}.ckpt outdir=${out_dir}/${f}
    fi
  done
  wait
  fi
  if [ $count2 -ne 0 ]
  then
  for f in ${query_dir}/16*.fa; do
    f=$(basename -- "$f")
    f="${f%.*}"
    if [ -z "${debug+x}" ]
    then
      depp_distance.py query_seq_file=${query_dir}/${f}.fa backbone_seq_file=${accessory_dir}/${f}_a.fasta model_path=${accessory_dir}/${f}.ckpt outdir=${out_dir}/${f} replicate=True> /dev/null 2>&1
    else
      depp_distance.py query_seq_file=${query_dir}/${f}.fa backbone_seq_file=${accessory_dir}/${f}_a.fasta model_path=${accessory_dir}/${f}.ckpt outdir=${out_dir}/${f} replicate=True
    fi
  done
  wait
  fi
fi
echo "finish distance matrices!"
echo "placing queries..."

mkdir ${out_dir}/summary > /dev/null 2>&1
if [ $count1 -ne 0 ]
then
  if [ -z "${aligned+x}" ]
    then
    for f in ${query_dir}/p*aligned.fa; do
        f=$(basename -- "$f")
        f="${f%_*}"
        run_apples.py -d ${out_dir}/${f}/depp.csv -t ${accessory_dir}/wol.nwk -o ${out_dir}/summary/${f}_placement.jplace -f 0.25 -b 25 > /dev/null 2>&1
    done
    wait
  else
    for f in ${query_dir}/p*.fa; do
        f=$(basename -- "$f")
        f="${f%.*}"
        run_apples.py -d ${out_dir}/${f}/depp.csv -t ${accessory_dir}/wol.nwk -o ${out_dir}/summary/${f}_placement.jplace -f 0.25 -b 25 > /dev/null 2>&1
    done
    wait
  fi
  agg_dist.py -o ${out_dir} -a ${accessory_dir} -p p
  run_apples.py -d ${out_dir}/summary/marker_genes_dist.csv -t ${accessory_dir}/wol.nwk -o ${out_dir}/summary/marker_genes_placement.jplace -f 0.25 -b 25 > /dev/null 2>&1
fi
if [ $count2 -ne 0 ]
then
  if [ -z "${aligned+x}" ]
  then
    for f in ${query_dir}/16*aligned.fa; do
        f=$(basename -- "$f")
        f="${f%_*}"
        run_apples.py -d ${out_dir}/${f}/depp.csv -t ${accessory_dir}/wol.nwk -o ${out_dir}/summary/${f}_placement.jplace -f 0.25 -b 25 > /dev/null 2>&1
        if [ $count1 -ne 0 ]
        then
        merge_json.py --out-dir ${out_dir}
        fi
    done
    wait
  else
    for f in ${query_dir}/16*.fa; do
        f=$(basename -- "$f")
        f="${f%.*}"
        run_apples.py -d ${out_dir}/${f}/depp.csv -t ${accessory_dir}/wol.nwk -o ${out_dir}/summary/${f}_placement.jplace -f 0.25 -b 25 > /dev/null 2>&1
        if [ $count1 -ne 0 ]
        then
        merge_json.py --out-dir ${out_dir}
        fi
    done
    wait
  fi
fi

gappa examine graft --jplace-path ${out_dir}/summary/ --out-dir ${out_dir}/summary/ --allow-file-overwriting > /dev/null 2>&1
echo "finish queries placement!"
