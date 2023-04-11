# $1 gene comb
# $2 gene num
line=${1}
echo $line
echo `echo $line | sed "s/,/_/g"`
new_dir=build_tree/${2}_more_repr/`echo $line | sed "s/,/_/g"`
filelist=`echo "${line}," | sed "s/p/build_tree\/p/g" | sed "s/,/\/seq.fa /g"`
echo $filelist
#rm -rf ${new_dir}/
mkdir -p ${new_dir}/
new_dir=${new_dir}/
IFS=',' read -r -a genes <<< "$line"
prob=""
names=""
for gene in "${genes[@]}";
do
	prob="${prob} build_tree/${gene}/build_tree/depp_dist_add_repr/class_prob.pt"
	names="${names} build_tree/${gene}/build_tree/depp_dist_add_repr/query_seq_names.pt"
done
mkdir ${new_dir}/assign_query
python scripts/select_cluster_by_prob.py --prob ${prob} --names ${names} --outdir ${new_dir}/assign_query
mkdir ${new_dir}/test_seqs
IFS=',' read -r -a genes <<< "$line"
for gene in "${genes[@]}";
do
	mkdir ${new_dir}/test_seqs/${gene}
	for c in {3,12};
	do
		if [ ! -f "${new_dir}/assign_query/${c}.txt" ];
		then
			continue
		fi
		rm ${new_dir}/tmp_labels
		cat ${new_dir}/assign_query/${c}.txt >> ${new_dir}/tmp_labels
		echo "" >> ${new_dir}/tmp_labels
		cat build_tree/${gene}/test_labels/${c}.txt >> ${new_dir}/tmp_labels
		echo "" >> ${new_dir}/tmp_labels
		cat build_tree/${gene}/add_repr_label_4k/${c}.txt >> ${new_dir}/tmp_labels
		sed -r -i '/^\s*$/d' ${new_dir}/tmp_labels
 		python ~/tool/grep_seq.py --infile build_tree/${gene}/seq.fa --outfile ${new_dir}/test_seqs/${gene}/${c}.fa --seqnames ${new_dir}/tmp_labels
 		python ~/tool/grep_seq.py --infile build_tree/${gene}/query.fa --outfile ${new_dir}/test_seqs/${gene}/query${c}.fa --seqnames ${new_dir}/assign_query/${c}.txt
#		cat ${new_dir}/test_seqs/${gene}/query${c}.fa build_tree/${gene}/test_seqs_add_repr/${c}.fa > ${new_dir}/test_seqs/${gene}/${c}.fa
#    cat ${new_dir}/test_seqs/${gene}/query${c}.fa build_tree/${gene}/test_seqs/${c}.fa > ${new_dir}/test_seqs/${gene}/${c}.fa
	done
	source activate cluster_depp
	for c in {3,12};
	do
		depp_distance.py seqdir=${new_dir}/test_seqs/${gene} query_seq_file=${new_dir}/test_seqs/${gene}/query${c}.fa model_path=build_tree/${gene}/model_build_tree_add_repr_multi/cluster-depp.pth outdir=${new_dir}/test_seqs/${gene} use_cluster=${c}
		python ~/tool/get_tree_dist.py --treefile
		python ~/tool/extend_csv.py --infile2 build_tree/${gene}/build_tree/backbone${c}.csv --infile1 ${new_dir}/test_seqs/${gene}/depp${c}.csv --outfile ${new_dir}/test_seqs/${gene}/dist${c}.csv  #build_tree/${gene}/rep${rep}/build_tree/dist${c}.csv
		python ~/tool/csv_to_phylip.py --infile ${new_dir}/test_seqs/${gene}/dist${c}.csv --outfile ${new_dir}/test_seqs/${gene}/dist${c}.phylip  #build_tree/${gene}/rep${rep}/build_tree/dist${c}.phylip
		mv ${new_dir}/test_seqs/${gene}/dist${c}.csv ${new_dir}/test_seqs/${gene}/dist.backbone_treedist${c}.csv
	done
done
for c in {3,12};
do
	mkdir -p ${new_dir}/build_tree
	IFS=',' read -r -a genes <<< "$line"
	distfile=""
	for gene in "${genes[@]}";
	do
		distfile="${distfile} ${new_dir}/test_seqs/${gene}/dist.backbone_treedist${c}.csv"
	done
	echo distfile $distfile
	python ~/tool/agg_dist.py --filelist $distfile --outfile ${new_dir}/build_tree/dist${c}.csv
	python ~/tool/csv_to_phylip.py --infile ${new_dir}/build_tree/dist${c}.csv --outfile ${new_dir}/build_tree/dist${c}.phylip
done
conda deactivate
for i in {3,12};
do
	echo "2"
	if [ ! -f "${new_dir}/assign_query/${i}.txt" ];
	then
		continue
	fi
	cnt=0
	fastme -i ${new_dir}/build_tree/dist${i}.phylip -o ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.${i}.nwk --nni -s -m B --output_info=${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.${i}.log
#	python scripts/add_polytomy.py --tree ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.${i}.nwk --prob ${new_dir}/assign_query/${i}.json --outdir ${new_dir}/build_tree --prefix ${i}
done
cat ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme*nwk.mapping > ${new_dir}/build_tree/mapping.txt
rm ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.trees
for i in ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme*nwk;
do
  cat ${i} >> ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.trees
  echo "" >> ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.trees
done
for i in 0;
do
	cat backbone.nwk >> ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.trees
done
nw_labels -I backbone.nwk > ${new_dir}/build_tree/backbone.label
echo "" >> ${new_dir}/build_tree/mapping.txt
paste -d ' ' ${new_dir}/build_tree/backbone.label ${new_dir}/build_tree/backbone.label >> ${new_dir}/build_tree/mapping.txt
sed -r -i '/^\s*$/d' ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.trees
sed -r -i '/^\s*$/d' ${new_dir}/build_tree/mapping.txt
#java -jar -Xmx160g ~/software/astral.5.6.9/Astral/astral.5.6.9.jar -i ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.trees -a ${new_dir}/build_tree/mapping.txt -o ${new_dir}/out.tree 2> ${new_dir}/astral.log
#~/software/ASTER/bin/astral -t 8 -i ${new_dir}/build_tree/fastme.newhyparam.tree_dist.balme.trees -o ${new_dir}/out_rep6.tree 2> ${new_dir}/astral.log.rep6
