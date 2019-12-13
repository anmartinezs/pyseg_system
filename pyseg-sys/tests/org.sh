#!/bin/bash

PYTHONPATH="${PYTHONPATH}:$PWD/.."

echo $PYTHONPATH

cp ../../data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto/fil_mb_sources_to_cyto_targets_net_parts.star ../../data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto/0_fil_mb_sources_to_cyto_targets_net_parts.star
cp ../../data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen/fil_mb_sources_to_lumen_targets_net_parts.star ../../data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen/1_fil_mb_sources_to_lumen_targets_net_parts.star

test_path=$PWD
cd ../tutorials/exp_ssmb/org/

printf -v cmd "%s" "python ltomos_generator.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/ltomos_generator.log

printf -v cmd "%s" "python uni_1st_analysis.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/uni_1st_analysis.log

printf -v cmd "%s" "python uni_2nd_analysis.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/uni_2nd_analysis.log

printf -v cmd "%s" "python bi_2nd_analysis.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/bi_2nd_analysis.log

cd $test_path