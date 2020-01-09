#!/bin/bash
set -ex

## Installing third parties packages

# Install DisPerSe (modified version for PySeg v1.0)
cd install/disperse/0.9.24_pyseg_gcc7
chmod u+x install.sh
./install.sh

# Install graph-tool (Anaconda)
# conda install -c conda-forge graph-tool
# # Install graph-tool (native)
# mkdir -p ./soft/graph-tool
# pushd install/graph-tool/2.2.44
# tar jxvf sources/graph-tool-2.2.44.tar.bz2 
# cd graph-tool-2.2.44
# ./configure --prefix=$sys_path/soft/graph-tool --disable-sparsehash
# make && make check && make install
# popd
