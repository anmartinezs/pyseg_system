#!/bin/bash

# Creating directries for intallation
#no final slash !!
targetFold=$PWD/../../../soft/graph-tool/2.2.44
buildFold=$targetFold/build
pythonFold=$targetFold/python

if [ -d "$targetFold" ]; then
  echo "removing $targetFold before installing !!"
  rm -r $targetFold/*
fi
mkdir $targetFold
mkdir $buildFold
mkdir $pythonFold

cp sources/graph-tool-2.2.44.tar.bz2 $targetFold
cd $targetFold
bunzip2 graph-tool-2.2.44.tar.bz2
tar -xvf graph-tool-2.2.44.tar
cd graph-tool-2.2.44

# Set environment
export CC=/usr/bin/gcc-4.8
export CXX=/usr/bin/g++-4.8
export MPI_CC=/usr/bin/gcc-4.8
export MPI_CXX=/usr/bin/g++-4.8
export OMPI_MPICXX=/usr/bin/g++-4.8

# CGAL Requirements
export CPPFLAGS="-I"$PWD"/../../../cgal/4.7/build/include -I"$PWD"/../../../cgal/4.7/infrstr/include"
export LDFLAGS="-L"$PWD"/../../../cgal/4.7/build/lib -L/../../../cgal/4.7/infrstr/lib"

# echo $CPPFLAGS
# echo $LDFLAGS

# Configuring
# ./configure --prefix=$buildFold --disable-sparsehash --disable-cairo --with-python-module-path=$pythonFold
./configure --prefix=$buildFold --disable-sparsehash --with-python-module-path=$pythonFold
# Installation
make
make check
make install prefix=$buildFold
