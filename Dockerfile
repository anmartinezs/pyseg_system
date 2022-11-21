# Copyright 2022 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

FROM ubuntu:20.04

WORKDIR /usr/local/pyseg_system
COPY . .

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update && \
    apt-get install -y \
      "gnupg2" \
      "ca-certificates" && \
    echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main" | tee -a /etc/apt/sources.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      "build-essential" \
      "bzip2" \
      "cmake" \
      "gcc" \
      "libboost-all-dev" \
      "libboost-python-dev" \
      "libcairomm-1.0-dev" \
      "libcgal-dev" \
      "libcgal-qt5-dev" \
      "libglu1-mesa" \
      "libgmp-dev" \
      "libgsl-dev" \
      "libmgl-dev" \
      "libxt6" \
      "m4" \
      "make" \
      "pkg-config" \
      "python-cairo-dev" \
      "unzip" \
      "zip" \
      "git" \
      "wget" \
      "python3-graph-tool" && \
    apt-get update && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7 && \
    mkdir -p /usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7/build && \
    mkdir -p /usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7/infrstr
ENV MAKEFLAGS="-j$(nproc)"
ARG MAKEFLAGS="-j$(nproc)"
RUN cp /usr/local/pyseg_system/sys/install/disperse/0.9.24_pyseg_gcc7/sources/disperse_v0.9.24_pyseg_gcc7.tar.gz /usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7 && \
    cp /usr/local/pyseg_system/sys/install/disperse/0.9.24_pyseg_gcc7/sources/cfitsio_3.380.tar.gz /usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7 && \
    cd /usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7 && \
    tar zxvf disperse_v0.9.24_pyseg_gcc7.tar.gz && tar zxvf cfitsio_3.380.tar.gz && \
    cd cfitsio && ./configure --enable-shared --enable-static --prefix=/usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7/infrstr && \
    make && make install && \
    cd /usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7/disperse/ && \
    cmake . -DCMAKE_INSTALL_PREFIX=/usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7/build -DCFITSIO_DIR=/usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7/infrstr && \
    make && make install

ENV PATH="/usr/local/miniconda3/bin:${PATH}"
ARG PATH="/usr/local/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /usr/local/miniconda3 \
    && rm -f miniconda.sh
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN conda install --yes \
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
      future
RUN conda clean --yes --all

ENV PATH="/usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7/build/bin:${PATH}"
ARG PATH="/usr/local/pyseg_system/sys/soft/disperse/0.9.24_pyseg_gcc7/build/bin:${PATH}"

ENV PYTHONPATH="/usr/local/pyseg_system/code:${PYTHONPATH}"
ARG PYTHONPATH="/usr/local/pyseg_system/code:${PYTHONPATH}"
