#!/bin/bash
set -ex

# Install and set first the solver to accelarte package installation
source $PWD/soft/miniconda3/bin/activate
conda install conda-libmamba-solver --yes \
  && conda config --set solver libmamba \
  && conda install --yes --solver=libmamba \
  --channel conda-forge --channel anaconda python=3.7 \
  opencv=4.2.0 \
  graph-tool=2.29 \
  future=0.18.2=py37_0 \
  pip=23.3 \
  && conda clean --yes --all \
  && pip install "setuptools<58" \
  && pip install beautifulsoup4==4.9.3 \
  lxml==4.6.3 \
  pillow==6.2.2 \
  pywavelets==1.1.1 \
  pyfits==3.5 \
  scikit-image==0.14.5 \
  scikit-learn==0.20.4 \
  scikit-fmm==2021.2.2 \
  scipy==1.2.1 \
  vtk==8.1.2 \
  astropy==4.1 \
  imageio==2.9.0 \

