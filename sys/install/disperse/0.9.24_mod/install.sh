#!/bin/bash

#no final slash !!
targetFold=/fs/pool/pool-bmsan-apps/antonio/sys/soft/disperse/0.9.24


if [ -d "$targetFold" ]; then
  echo "error: remove $targetFold before installing !!"
  exit 1
fi
mkdir $targetFold
buildFold=$targetFold/build
infrFold=${targetFold}/infrstr
mkdir $infrFold
mkdir $buildFold

# Setting up environment variables
export CC=/usr/bin/gcc-4.8
export CXX=/usr/bin/g++-4.8
export MPI_CC=/usr/bin/gcc-4.8
export MPI_CXX=/usr/bin/g++-4.8
export OMPI_MPICXX=/usr/bin/g++-4.8
CGAL_DIR=/fs/pool/pool-bmsan-apps/antonio/sys/soft/cgal/4.7/build/lib/CGAL
CFITSIO_DIR=$infrFold

# Copy already downloaded package
cp ./sources/disperse_v0.9.24.tar.gz $targetFold
cp ./sources/cfitsio_3.380.tar.gz $targetFold
cd $targetFold
gunzip disperse_v0.9.24.tar.gz
tar xvf disperse_v0.9.24.tar

# CFitsIO
gunzip cfitsio_3.380.tar.gz
tar xvf cfitsio_3.380.tar
cd cfitsio
./configure --enable-shared --enable-static --prefix=$infrFold
make && make install
cd $targetFold/disperse/

# Configuration
cmake . -DCMAKE_INSTALL_PREFIX=$buildFold -DCGAL_DIR=$CGAL_DIR -DCFITSIO_DIR=$CFITSIO_DIR

# Installation
make && make install
