#!/bin/bash
set -ex

## Create a file for the environmental variables
if [ -f soft/bashrc_pyseg_sys ] 
then
    rm soft/bashrc_pyseg_sys
fi
printf -v date '%(%d.%m.%Y)T\n' -1 
printf "\n## PySeg system "$date" \n" >> soft/bashrc_pyseg_sys
printf "\n## DisPerSe \n" >> soft/bashrc_pyseg_sys
printf "export PATH="$PWD"/soft/disperse/0.9.24_pyseg_gcc7/build/bin:""$""PATH" >> soft/bashrc_pyseg_sys
printf "\n## Conda python \n" >> soft/bashrc_pyseg_sys
printf "export PATH="$PWD"/soft/miniconda2/bin:""$""PATH" >> soft/bashrc_pyseg_sys
printf "\n# PySeg python code \n" >> soft/bashrc_pyseg_sys
printf "export PYTHONPATH="$PWD"/../pyseg-sys:""$""PYTHONPATH" >> soft/bashrc_pyseg_sys
printf "\n" >> soft/bashrc_pyseg_sys

## Add entry to .bashrc
printf "\n## PySeg system "$date" \n" >> ~/.bashrc
printf "\n" >> ~/.bashrc
printf "source "$PWD"/soft/bashrc_pyseg_sys" >> ~/.bashrc
printf "\n\n" >> ~/.bashrc
exec bash
