from setuptools import setup,find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
	name='depp',    # This is the name of your PyPI-package.
        version='0.3.1',    # Update the version number for new releases
        scripts=['train_depp.py',
                 'depp/depp_distance.py',
                 'agg_dist.py',
                 'wol_placement.sh',
                 'run_upp.sh',
                 'merge_json.py',
		 'train_depp_recon.py',
                 'depp-place-rRNA.sh',
                 'depp-place-rRNA-one-type.sh',
                 'comb_json.py',
                 'seq_sep.py',
                 'depp/filter_by_entropy_gap.sh',
                 'depp/grep_seq.py',
                 'depp/get_names_by_entropy_gap.py',
                 'depp/grep_jplace.py',
                 'depp/count_gapped_ratio.py',
                 'depp/set_bl_one.py',
                 'depp/grep_seq_group.py',
                 'depp/get_tree_dist.py',
                 'depp/select_species_by_distance.py',
                 'train_cluster_depp.sh'], # The name of your scipt, and also the command you'll be using for calling it
        description='DEPP: Deep Learning Enables Extending Species Trees using Single Genes',
        long_description='DEPP is a deep-learning-based tool for phylogenetic placement.'
                         'Output of the tool is the distance matrix between the query sequences and the backbone sequences',
        long_description_content_type='text/plain',
        url='https://github.com/yueyujiang/DEPP',
        author='Yueyu Jiang',
        author_email='y5jiang@ucsd.edu',
        packages=find_packages(),
        zip_safe = False,
        install_requires=[
			'numpy==1.22.3',
			'treeswift==1.1.19',
			'torch==1.11.0',
			'pandas==1.4.2',
			'pytorch-lightning==1.5.4',
			'biopython==1.79',
			'omegaconf==2.1.0',
			'apples',
			'scipy==1.7.3',
			'dendropy==4.5.2'
			],
        include_package_data=True
)
