#!/bin/bash
set -ex

#no final slash !!
targetFold=$PWD/../../../soft/disperse/latest

if [ -d "$targetFold" ]; then
  echo "removing $targetFold before installing !!"
  rm -r $targetFold/*
fi
mkdir -p $targetFold
buildFold=$targetFold/build
infrFold=${targetFold}/infrstr
mkdir $infrFold
mkdir $buildFold

# Setting up environment variables
export MAKEFLAGS="-j$(nproc)"

# Copy already downloaded package
pwd
cp ./sources/disperse_latest.tar.gz $targetFold
cp ./sources/cfitsio_3.380.tar.gz $targetFold
cd $targetFold
tar zxvf disperse_latest.tar.gz
mv DisPerSE-master disperse
tar zxvf cfitsio_3.380.tar.gz

# CFitsIO
cd cfitsio
./configure --enable-shared --enable-static --prefix=$infrFold
make && make install

# Configuration
cd $targetFold/disperse/
# cmake . -DCMAKE_INSTALL_PREFIX=$buildFold -DCGAL_DIR=$CGAL_DIR -DCFITSIO_DIR=$CFITSIO_DIR
cmake . -DCMAKE_INSTALL_PREFIX=$buildFold -DCFITSIO_DIR=$infrFold

# Installation
make && make install
