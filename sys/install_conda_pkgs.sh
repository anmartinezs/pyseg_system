#!/bin/bash
set -ex

## Installing conda packages (requires Anaconda with Python 3)
$PWD/soft/miniconda3/bin/conda install --yes \
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
    && $PWD/soft/miniconda3/bin/conda clean --yes --all
    
# ## Installing conda VTK package (it must be installed after graph-tool to avoid conficts)
# $PWD/soft/miniconda3/bin/conda install --yes \
#       --channel conda-forge \
#       vtk \
#     && $PWD/soft/miniconda3/bin/conda clean --yes --all
