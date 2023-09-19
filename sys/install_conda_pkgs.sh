#!/bin/bash
set -ex

if [ -x "$(command -v conda)" ]; then 
  clean_exe="conda"
elif [ -x "$(command -v $PWD/soft/miniconda3/bin/conda)" ]; then 
  clean_exe="$PWD/soft/miniconda3/bin/conda"
else 
  echo "ERROR!! Conda not installed! Exiting..."
  exit
fi

if [ -x "$(command -v mamba)" ]; then 
  install_exe="mamba"
else
  install_exe="$clean_exe"
fi

## Installing conda packages (requires Anaconda with Python 3)
$install_exe install --yes \
      --channel conda-forge \
      graph-tool \
      beautifulsoup4 \
      lxml \
      numpy \
      pillow \
      scikit-image \
      scikit-learn \
      scipy \
      vtk \
      scikit-fmm \
      astropy \
      pytest \
      opencv \
      future \
    && $clean_exe clean --yes --all
    
# ## Installing conda VTK package (it must be installed after graph-tool to avoid conficts)
# $PWD/soft/miniconda3/bin/conda install --yes \
#       --channel conda-forge \
#       vtk \
#     && $PWD/soft/miniconda3/bin/conda clean --yes --all
