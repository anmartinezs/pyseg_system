#!/bin/bash
set -ex

## Installing Anaconda
if [ -d "sys/soft/minconda2" ] 
then
    rm -r soft/miniconda2
else
    mkdir -p soft
fi
sysPath=$PWD
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh
bash ./miniconda.sh -p $sysPath/soft/miniconda2


