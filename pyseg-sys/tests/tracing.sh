#!/bin/bash

PYTHONPATH="${PYTHONPATH}:$PWD/../"

test_path=$PWD
cd ../tutorials/exp_ssmb/tracing/

printf -v cmd "%s" "python mb_graph_cyto.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/mb_graph_cyto.log

printf -v cmd "%s" "python mb_graph_lumen.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/mb_graph_lumen.log

printf -v cmd "%s" "python mb_fils_network_cyto.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/mb_fils_network_cyto.log

printf -v cmd "%s" "python mb_fils_network_lumen.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/mb_fils_network_lumen.log

printf -v cmd "%s" "python mb_picking_cyto.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/mb_picking_cyto.log

printf -v cmd "%s" "python mb_picking_lumen.py"
echo $cmd
$cmd | tee $PWD/../../../tests/results/mb_picking_lumen.log

cd $test_path