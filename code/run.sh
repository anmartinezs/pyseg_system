#!/bin/bash
set -ex

# Setting system global variables
ls /deps/*
export PATH=/deps/disperse/bin:$PATH
export PYTHONPATH=/code/python:$PYTHONPATH

# Directories for output data
mkdir -p tests/results
# mkdir -p /data/synthetic_test/out
mkdir -p /data/tutorials/exp_ssmb/klass/out
mkdir -p /data/tutorials/exp_ssmb/tracing/mb_single/graph/lumen/disperse
mkdir -p /data/tutorials/exp_ssmb/tracing/mb_single/graph/cyto/disperse
mkdir -p /data/tutorials/exp_ssmb/tracing/mb_single/fils/cyto
mkdir -p /data/tutorials/exp_ssmb/tracing/mb_single/fils/lumen
mkdir -p /data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto
mkdir -p /data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen
mkdir -p /data/tutorials/exp_ssmb/stat/in
mkdir -p /data/tutorials/exp_ssmb/stat/ltomos/test
mkdir -p /data/tutorials/exp_ssmb/stat/out

# Running the tests
cd tests
chmod u+x *.sh

./tracing_grid.sh
mkdir -p /results/tracing_grid
cp /code/tests/results/synthetic_test.log /results/tracing_grid/tracing_grid.log
cp /data/synthetic_grid/test_grid_conn.png /results/tracing_grid/test_grid_conn.png
cd /code/tests

./classification.sh
mkdir -p /results/klass
cp /code/tests/results/plane_align_class_test.log /results/klass/klass.log
cd /data/tutorials/exp_ssmb/klass/
zip -r klass_out.zip ./*
mv klass_out.zip /results/klass/
cd /code/tests

./tracing.sh
# Graph computation result
mkdir -p /results/tracing/graph
cp /code/tests/results/mb_graph_cyto.log /results/tracing/graph/
cd /data/tutorials/exp_ssmb/tracing/mb_single/graph/cyto/
zip -r mb_graph_cyto_out.zip ./*
mv mb_graph_cyto_out.zip /results/tracing/graph/
cp /code/tests/results/mb_graph_lumen.log /results/tracing/graph/
cd /data/tutorials/exp_ssmb/tracing/mb_single/graph/lumen/
zip -r mb_graph_lumen_out.zip ./*
mv mb_graph_lumen_out.zip /results/tracing/graph/
# Filaments extraction results
mkdir -p /results/tracing/fils
cp /code/tests/results/mb_fils_network_cyto.log /results/tracing/fils/
cd /data/tutorials/exp_ssmb/tracing/mb_single/fils/cyto/
zip -r mb_fils_cyto_out.zip ./*
mv mb_fils_cyto_out.zip /results/tracing/fils/
cp /code/tests/results/mb_fils_network_lumen.log /results/tracing/fils/
cd /data/tutorials/exp_ssmb/tracing/mb_single/fils/lumen/
zip -r mb_fils_lumen_out.zip ./*
mv mb_fils_lumen_out.zip /results/tracing/fils/
# Particles picking results
mkdir -p /results/tracing/pick
cp /code/tests/results/mb_picking_cyto.log /results/tracing/pick/
cd /data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto/
zip -r mb_pick_cyto_out.zip ./*
mv mb_pick_cyto_out.zip /results/tracing/pick/
cp /code/tests/results/mb_picking_lumen.log /results/tracing/pick/
cd /data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen/
zip -r mb_pick_lumen_out.zip ./*
mv mb_pick_lumen_out.zip /results/tracing/pick/
cd /code/tests

./org.sh
mkdir -p /results/stat
cd /data/tutorials/exp_ssmb/stat/out/uni_1st
zip -r uni_1st_out.zip ./*
mv uni_1st_out.zip /results/stat
cd /data/tutorials/exp_ssmb/stat/out/uni_2nd
zip -r uni.zip ./*
mv uni.zip /results/stat
cd /data/tutorials/exp_ssmb/stat/out/bi_2nd
zip -r bi.zip ./*
mv bi.zip /results/stat
cp /code/tests/results/ltomos_generator.log /results/stat
cp /code/tests/results/uni_1st_analysis.log /results/stat
cp /code/tests/results/uni_2nd_analysis.log /results/stat
cp /code/tests/results/bi_2nd_analysis.log /results/stat
