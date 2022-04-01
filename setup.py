from setuptools import setup,find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
	name='depp_test',    # This is the name of your PyPI-package.
        version='0.0.3',    # Update the version number for new releases
        scripts=['train_depp.py',
                 #'distance_depp.sh',
                 'depp/depp_distance.py',
                 'depp/distance_correction.py',
                 'agg_dist.py',
                 'wol_placement.sh',
                 'run_upp.sh',
                 'merge_json.py',
		 'train_depp_recon.py',
                 'depp-place-rRNA.sh',
                 'depp-place-rRNA-one-type.sh',
                 'comb_json.py',
                 'seq_sep.py'], # The name of your scipt, and also the command you'll be using for calling it
        description='DEPP: Deep Learning Enables Extending Species Trees using Single Genes',
        long_description='DEPP is a deep-learning-based tool for phylogenetic placement.'
                         'Output of the tool is the distance matrix between the query sequences and the backbone sequences',
        long_description_content_type='text/plain',
        url='https://github.com/yueyujiang/DEPP',
        author='Yueyu Jiang',
        author_email='y5jiang@ucsd.edu',
        packages=find_packages(),
        zip_safe = False,
        install_requires=['numpy==1.19.1',
			'treeswift==1.1.19',
			#'torch==1.7.1',
			'torch>=1.3,<=1.7.1',
			'pandas==1.3.0',
			'pytorch-lightning==1.0.5',
			'biopython==1.79',
			'omegaconf==2.1.0',
			'apples',
			'dendropy==4.5.2'
			],
        include_package_data=True
)
