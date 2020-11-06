while getopts q:b:m:t:o: flag
do
    case "${flag}" in
	q) query_sequence=${OPTARG};;
	b) backbone_sequence=${OPTARG};;
	m) model_path=${OPTARG};;
	t) backbone_tree=${OPTARG};;
	o) outdir=${OPTARG};;
    esac
done
python depp_distance.py query_seq_file=$query_sequence model_path=$model_path backbone_seq_file=$backbone_sequence outdir=$outdir
echo "distance correcting..."
mkdir $outdir/depp_tmp
run_apples.py -d "./${ourdir}/depp_distance/depp.csv" -t "./${backbone_tree}" -o "./${outdir}/depp_tmp/tmp.jplace"
gappa examine graft --jplace-path "./${outdir}/depp_tmp/tmp.jplace" --out-dir "./${outdir}/depp_tmp" --allow-file-overwriting > /dev/null 2>&1 
perl -ne 'if(/^>(\S+)/){print "$1\n"}' $query_sequence > $outdir/depp_tmp/seq_name.txt
while read p; do
    sed -e's/);/,XXXXX:0)ROOT:0;/g' "${outdir}/depp_tmp/tmp.newick"|nw_reroot - $p| nw_clade - ROOT|nw_labels -I -|grep -v XXXXX > "${outdir}/depp_tmp/${p}_leaves.txt";
done < $outdir/depp_tmp/seq_name.txt
python distance_correction.py outdir=$outdir backbone_tree=$backbone_tree
echo "finish correction!"
run_apples.py -d "./${ourdir}/depp_distance/depp_correction.csv" -t "./${backbone_tree}" -o "./${outdir}/depp_tmp/tmp.jplace"
rm -rf "./${outdir}/depp_tmp/"
