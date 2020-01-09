#!/bin/bash

#no final slash !!
targetFold=$PWD/../../../soft/vtk/6.3.0


if [ -d "$targetFold" ]; then
  echo "removing $targetFold before installing !!"
  rm -r $targetFold/*
fi
mkdir $targetFold

# Copy already downloaded package
cp ./sources/VTK-6.3.0.tar.gz $targetFold
cd $targetFold
gunzip VTK-6.3.0.tar.gz
tar xvf VTK-6.3.0.tar
cd VTK-6.3.0/

buildFold=$targetFold/build
mkdir $buildFold

# Setting up environment variables
export CC=/usr/bin/gcc-4.8
export CXX=/usr/bin/g++-4.8
export MPI_CC=/usr/bin/gcc-4.8
export MPI_CXX=/usr/bin/g++-4.8
export OMPI_MPICXX=/usr/bin/g++-4.8

# Configuration
cmake . -DCMAKE_INSTALL_PREFIX=$buildFold -DVTK_WRAP_PYTHON=ON

# Installation
make && make install
