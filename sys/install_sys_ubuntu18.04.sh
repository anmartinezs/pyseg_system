#!/bin/bash
set -ex

## Installing Ubuntu standard packages
# sudo apt-get update \
#     && sudo apt-get install -y --no-install-recommends \
#       "build-essential" \
#       "bzip2" \
#       "cmake" \
#       "gcc" \
#       "libboost-all-dev" \
#       "libboost-python-dev" \
#       "libcairomm-1.0-dev" \
#       "libcgal-dev" \
#       "libcgal-qt5-dev" \
#       "libglu1-mesa" \
#       "libgmp-dev" \
#       "libgsl-dev" \
#       "libmgl-dev" \
#       "libxt6" \
#       "m4" \
#       "make" \
#       "pkg-config" \
#       "python-cairo-dev" \
#       "unzip" \
#       "zip" \
#       "git" \

## Installing conda packages (required Ancaonda with Python 2.7)
conda install --yes \
      --channel menpo \
      --channel jochym \
      --channel anaconda \
      beautifulsoup4==4.6.1 \
      lxml==4.2.4 \
      numpy==1.15.0 \
      opencv==2.4.11 \
      pillow==5.2.0 \
      pyfits==3.4.0 \
      scikit-image==0.14.0 \
      scikit-learn==0.19.1 \
      scipy==1.1.0 \
      vtk==6.3.0 \
      scikit-fmm \
      graph-tool \
    && conda clean --yes --all

## Installing local system

# Install DisPerSe (cfitsio needed)
cd sys
sys_path=$PWD
export MAKEFLAGS="-j$(nproc)"
mkdir -p ./soft/disperse ./soft/cfitsio
pushd install/disperse/0.9.24_mod_ubuntu18.04
tar zxvf sources/cfitsio_3.380.tar.gz && cd cfitsio
./configure --enable-shared --enable-static --prefix=$sys_path/soft/cfitsio
make && make install
cd ..
# tar zxvf sources/disperse_v0.9.24_mod.tar.gz && cd disperse
cd /home/martinez/workspace/disperse_mod/disperse
cmake . -DCMAKE_INSTALL_PREFIX=$sys_path/soft/disperse -DCFITSIO_DIR=$sys_path/soft/cfitsio
make && make install
#popd

# Install graph-tool (Anaconda)
conda install -c conda-forge graph-tool
# # Install graph-tool (native)
# mkdir -p ./soft/graph-tool
# pushd install/graph-tool/2.2.44
# tar jxvf sources/graph-tool-2.2.44.tar.bz2 
# cd graph-tool-2.2.44
# ./configure --prefix=$sys_path/soft/graph-tool --disable-sparsehash
# make && make check && make install
# popd

## Adding environmental variables to .bashrc
export PATH=$sys_path/soft/disperse/bin:$PATH
export PYTHONPATH=$sys_path/../pyseg-sys/:$PYTHONPATH
