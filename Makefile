# allows running all the commands inside an operation in the same shell
.ONESHELL:
SHELL = /bin/bash

# creates and configures the environment.
env : 
        source /srv/conda/etc/profile.d/conda.sh
        conda env create -f environment.yml 
        conda activate notebook
        conda install ipykernel
        python -m ipykernel install --user --name make-env --display-name "IPython - Make"

# build the JupyterBook normally
.PHONY : html
html:
        jupyterbook build .
        

# clean up the `figures`, `audio`  and `_build` folders.
.PHONY : clean
clean:
        rm -rf figures/*
        rm -rf audio/*
        rm -rf _build/*