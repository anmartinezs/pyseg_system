#!/bin/bash

PYTHONPATH="${PYTHONPATH}:$PWD/../"

printf -v cmd "%s" "python ../tutorials/exp_ssmb/class/plane_align_class_test.py"
echo $cmd
$cmd | tee $PWD/results/plane_align_class_test.log