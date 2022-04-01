#FROM ubuntu:18.04
FROM --platform=linux/amd64 ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda install python==3.8.12 
RUN conda install -c bioconda gappa &\
    conda install -c bioconda newick_utils &\
    pip install depp_test==0.0.3

RUN conda install -c anaconda git

RUN cd ~ && git clone https://github.com/smirarab/sepp && cd sepp && python setup.py config && python setup.py install

RUN mkdir ~/pasta-code && cd ~/pasta-code && git clone https://github.com/smirarab/pasta.git && git clone https://github.com/smirarab/sate-tools-linux.git && cd pasta && python setup.py develop 

RUN cd ~/sepp && python setup.py upp

# RUN wget https://tera-trees.com/data/depp/latest/depp_env.yml && conda env create -f depp_env.yml && rm depp_env.yml

# ENV PATH /opt/conda/envs/depp_env/bin:$PATH
# ENV CONDA_DEFAULT_ENV depp_env

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "depp_env", "/bin/bash", "-c"]

# EXPOSE 5003
# The code to run when container is started:
# ENTRYPOINT ["conda", "run", "-n", "depp_env", "/bin/bash", "-c"]
