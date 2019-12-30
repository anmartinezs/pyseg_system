#!/bin/bash
set -ex

## Installing conda packages (requires Anaconda with Python 2.7)
conda install --yes \
      --channel menpo \
      --channel jochym \
      --channel anaconda \
      --channel conda-forge \
      beautifulsoup4 \
      lxml \
      numpy \
      opencv \
      pillow \
      pyfits \
      scikit-image \
      scikit-learn \
      scipy \
      vtk \
      scikit-fmm \
      graph-tool \
    && conda clean --yes --all
