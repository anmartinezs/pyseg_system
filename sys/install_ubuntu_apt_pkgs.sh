#!/bin/bash
set -ex

## Installing Ubuntu APT packages (required sudo privileges)
sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends \
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
      "python3-graph-tool" \

