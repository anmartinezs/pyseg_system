#!/bin/bash
set -ex

## Installing third parties packages

# Install DisPerSe (modified version for PySeg v1.0)
cd install/disperse/0.9.24_pyseg_gcc7
chmod u+x install.sh
./install.sh


