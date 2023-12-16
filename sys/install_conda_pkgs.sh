#!/bin/bash
set -ex

if [ -x "$(command -v conda)" ]; then 
  clean_exe="conda"
elif [ -x "$(command -v $PWD/soft/miniconda3/bin/conda)" ]; then 
  clean_exe="$PWD/soft/miniconda3/bin/conda"
  source $PWD/soft/miniconda3/bin/activate
else 
  echo "ERROR!! Conda not installed! Exiting..."
  exit
fi

if [ -x "$(command -v mamba)" ]; then 
  install_exe="mamba"
else
  install_exe="$clean_exe"
fi

$install_exe install conda-libmamba-solver --yes \
  && conda config --set solver libmamba
  
status_code=$?
if [[ $status_code -eq 0 ]] ; then
  solver_cmd="$install_exe install --yes --solver=libmamba"
else
  solver_cmd="$install_exe install --yes"
fi
  
$solver_cmd --channel conda-forge --channel anaconda \
  python=3.7 \
  opencv=4.2.0 \
  graph-tool=2.29 \
  future=0.18.2=py37_0 \
  pip=23.3 \
  && $install_exe clean --yes --all \
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
  && $clean_exe clean --yes --all

#      numpy \
#      pytest \
