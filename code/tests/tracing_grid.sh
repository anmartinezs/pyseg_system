#!/bin/bash

PYTHONPATH="${PYTHONPATH}:$PWD/../"

printf -v cmd "%s" "python ../pyseg/scripts/tests/mcf_synthetic_test.py do_short"
echo $cmd
$cmd | tee $PWD/results/synthetic_test.log