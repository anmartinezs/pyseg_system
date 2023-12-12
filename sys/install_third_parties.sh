#!/bin/bash
set -ex

## Installing third parties packages

# Install latest DisPerSe from source
wget --no-check-certificate https://github.com/thierry-sousbie/DisPerSE/archive/refs/heads/master.tar.gz
mv master.tar.gz install/disperse/latest/sources/disperse_latest.tar.gz
cd install/disperse/latest
chmod u+x install.sh
./install.sh


