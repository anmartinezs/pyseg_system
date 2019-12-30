## Clean data in output directories used for testing/tutorials
#!/bin/bash
set -ex

## TEST

rm -r code/tests/results/*

## TUTORIAL exp_ssmb

# tracing
rm -r data/tutorials/exp_ssmb/tracing/mb_single/graph/cyto/*
mkdir -p data/tutorials/exp_ssmb/tracing/mb_single/graph/cyto
rm -r data/tutorials/exp_ssmb/tracing/mb_single/graph/lumen/*
mkdir -p data/tutorials/exp_ssmb/tracing/mb_single/graph/lumen
rm -r data/tutorials/exp_ssmb/tracing/mb_single/fils/cyto/*
mkdir -p data/tutorials/exp_ssmb/tracing/mb_single/fils/cyto
rm -r data/tutorials/exp_ssmb/tracing/mb_single/fils/lumen/*
mkdir -p data/tutorials/exp_ssmb/tracing/mb_single/fils/lumen
rm -r data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto/*
mkdir -p data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto
rm -r data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen/*
mkdir -p data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen

# classification
rm -r data/tutorials/exp_ssmb/klass/out
mkdir data/tutorials/exp_ssmb/klass/out

# stat
rm -r data/tutorials/exp_ssmb/stat/ltomos/*
mkdir data/tutorials/exp_ssmb/stat/ltomos/test
mkdir data/tutorials/exp_ssmb/stat/ltomos/test/test_ssup_8
rm -r data/tutorials/exp_ssmb/stat/out
mkdir data/tutorials/exp_ssmb/klass/out

## TUTORIAL synth_sumb

rm -r data/tutorials/synth_sumb/trash/*
rm -r data/tutorials/synth_sumb/mics/*
rm -r data/tutorials/synth_sumb/segs/*
rm -r data/tutorials/synth_sumb/graphs/*
rm -r data/tutorials/synth_sumb/fils/out*
rm -r data/tutorials/synth_sumb/pick/out*
rm -r data/tutorials/synth_sumb/rec/*
rm -r data/tutorials/synth_sumb/class/*
rm -r data/tutorials/synth_sumb/rln/*
rm -r data/tutorials/synth_sumb/org/ltomos/*
rm -r data/tutorials/synth_sumb/org/uni_1st/*
rm -r data/tutorials/synth_sumb/org/uni_2nd/*
rm -r data/tutorials/synth_sumb/org/bi_2nd/*
