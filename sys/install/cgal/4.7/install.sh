#!/bin/bash

#no final slash !!
targetFold=$PWD/../../../soft/cgal/4.7


if [ -d "$targetFold" ]; then
  echo "removing $targetFold before installing !!"
  rm -r $targetFold/*
fi
mkdir $targetFold

# Copy already downloaded package
cp ./sources/CGAL-4.7.tar.gz $targetFold
cd $targetFold
gunzip CGAL-4.7.tar.gz
tar xvf CGAL-4.7.tar
cd CGAL-4.7/

buildFold=$targetFold/build
infrFold=${targetFold}/infrstr
mkdir $infrFold
mkdir $buildFold
cd $targetFold

#GMP
wget ftp://gcc.gnu.org/pub/gcc/infrastructure/gmp-4.3.2.tar.bz2
bunzip2 gmp-4.3.2.tar.bz2
tar xvf gmp-4.3.2.tar
cd gmp-4.3.2
./configure --enable-shared --enable-static --prefix=$infrFold
make && make install
cd ..


#MPFR
wget ftp://gcc.gnu.org/pub/gcc/infrastructure/mpfr-2.4.2.tar.bz2
bunzip2 mpfr-2.4.2.tar.bz2
tar xvf mpfr-2.4.2.tar
cd mpfr-2.4.2
./configure --enable-shared --enable-static --prefix=$infrFold --with-gmp=$infrFold
make && make install
cd ../CGAL-4.7/

cd $targetFold/CGAL-4.7/
# Setting up environment variables
export CC=/usr/bin/gcc-4.8
export CXX=/usr/bin/g++-4.8
export MPI_CC=/usr/bin/gcc-4.8
export MPI_CXX=/usr/bin/g++-4.8
export OMPI_MPICXX=/usr/bin/g++-4.8
LIB_GMP_DIR=$infrFold/lib
LIB_MPFR_DIR=$infrFold/lib
INCLUDE_GMP_DIR=$infrFold/include
INCLUDE_MPFR_DIR=$infrFold/include
LIB_GMP=$LIB_GMP_DIR/libgmp.so
LIB_MPFR=$LIB_MPFR_DIR/libmpfr.so

# Configuration
cmake . -DCMAKE_INSTALL_PREFIX=$buildFold -DGMP_LIBRARIES=$LIB_GMP -DGMP_INCLUDE_DIR=$INCLUDE_GMP_DIR -DMPFR_LIBRARIES=$LIB_MPFR -DMPFR_INCLUDE_DIR=$INCLUDE_MPFR_DIR

# Installation
make && make install
