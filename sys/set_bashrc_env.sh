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
#printf "\n## Conda python \n" >> soft/bashrc_pyseg_sys

#printf "export PATH="$PWD"/soft/miniconda3/bin:""$""PATH" >> soft/bashrc_pyseg_sys
if [ -x "$(command -v $PWD/soft/miniconda3/bin/conda)" ] && ! [ -x "$(command -v conda)" ]; then 
  printf "export PATH="$PWD"/soft/miniconda3/bin:""$""PATH" >> soft/bashrc_pyseg_sys
elif ! [ -x "$(command -v conda)" ]; then 
  echo "ERROR!! Conda not installed! Exiting..."
  exit
fi

printf "\n# PySeg python code \n" >> soft/bashrc_pyseg_sys
printf "export PYTHONPATH="$PWD"/../code:""$""PYTHONPATH" >> soft/bashrc_pyseg_sys
printf "\n" >> soft/bashrc_pyseg_sys

## Add entry to .bashrc
printf "\n## PySeg system "$date" \n" >> ~/.bashrc
printf "\n" >> ~/.bashrc
printf "export use_pyseg='source "$PWD"/soft/bashrc_pyseg_sys'" >> ~/.bashrc
printf "\n\n" >> ~/.bashrc
exec bash
