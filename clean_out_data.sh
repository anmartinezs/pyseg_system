## Clean data in output directories used for testing/tutorials
#!/bin/bash
set -x

## TEST
echo $PWD/
if [ -d $PWD/code/tests/results ]; then
	rm -r $PWD/code/tests/results/*
else
	mkdir $PWD/code/tests/results
fi

## TUTORIAL exp_ssmb

# tracing
if [ -d $PWD/data/tutorials/exp_ssmb/tracing/mb_single/graph/cyto ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/tracing/mb_single/graph/cyto/*
else
	mkdir -p $PWD/data/tutorials/exp_ssmb/tracing/mb_single/graph/cyto
fi
if [ -d $PWD/data/tutorials/exp_ssmb/tracing/mb_single/graph/lumen ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/tracing/mb_single/graph/lumen/*
else
	mkdir -p $PWD/data/tutorials/exp_ssmb/tracing/mb_single/graph/lumen
fi
if [ -d $PWD/data/tutorials/exp_ssmb/tracing/mb_single/fils/cyto ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/tracing/mb_single/fils/cyto/*
else
	mkdir -p $PWD/data/tutorials/exp_ssmb/tracing/mb_single/fils/cyto
fi
if [ -d $PWD/data/tutorials/exp_ssmb/tracing/mb_single/fils/lumen ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/tracing/mb_single/fils/lumen/*
else
	mkdir -p $PWD/data/tutorials/exp_ssmb/tracing/mb_single/fils/lumen
fi
if [ -d $PWD/data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto/*
else
	mkdir -p $PWD/data/tutorials/exp_ssmb/tracing/mb_single/pick/cyto
fi
if [ -d $PWD/data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen/*
else
	mkdir -p $PWD/data/tutorials/exp_ssmb/tracing/mb_single/pick/lumen
fi

# classification
if [ -d $PWD/data/tutorials/exp_ssmb/klass/out ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/klass/out
else
	mkdir $PWD/data/tutorials/exp_ssmb/klass/out
fi

# stat
if [ -d $PWD/data/tutorials/exp_ssmb/stat/ltomos ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/stat/ltomos/*
else
	mkdir -p $PWD/data/tutorials/exp_ssmb/stat/ltomos/test/test_ssup_8
fi
if [ -d $PWD/data/tutorials/exp_ssmb/stat/out ]; then
	rm -r $PWD/data/tutorials/exp_ssmb/stat/out/*
else
	mkdir -p $PWD/data/tutorials/exp_ssmb/klass/out
fi
mkdir

## TUTORIAL synth_sumb

if [ -d $PWD/data/tutorials/synth_sumb/trash ]; then
	rm -r $PWD/data/tutorials/synth_sumb/trash/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/trash
fi
if [ -d $PWD/data/tutorials/synth_sumb/mics ]; then
	rm -r $PWD/data/tutorials/synth_sumb/mics/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/mics
fi
if [ -d $PWD/data/tutorials/synth_sumb/segs ]; then
	rm -r $PWD/data/tutorials/synth_sumb/segs/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/segs
fi
if [ -d $PWD/data/tutorials/synth_sumb/graphs ]; then
	rm -r $PWD/data/tutorials/synth_sumb/graphs/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/graphs
fi
if [ -d $PWD/data/tutorials/synth_sumb/fils/out ]; then
	rm -r $PWD/data/tutorials/synth_sumb/fils/out/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/fils/out
fi
if [ -d $PWD/data/tutorials/synth_sumb/pick/out ]; then
	rm -r $PWD/data/tutorials/synth_sumb/pick/out/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/pick/out
fi
if [ -f $PWD/data/tutorials/synth_sumb/rec/mask_sph_130_60.mrc ]; then
	mv $PWD/data/tutorials/synth_sumb/rec/mask_sph_130_60.mrc $PWD/data/tutorials/synth_sumb/
fi
if [ -f $PWD/data/tutorials/synth_sumb/rec/wedge_130_60.mrc ]; then
	mv $PWD/data/tutorials/synth_sumb/rec/wedge_130_60.mrc $PWD/data/tutorials/synth_sumb/
fi
rm -r $PWD/data/tutorials/synth_sumb/rec/*
if [ -f $PWD/data/tutorials/synth_sumb/mask_sph_130_60.mrc ]; then
	mv $PWD/data/tutorials/synth_sumb/mask_sph_130_60.mrc $PWD/data/tutorials/synth_sumb/rec/
fi
if [ -f $PWD/data/tutorials/synth_sumb/wedge_130_60.mrc ]; then
	mv $PWD/data/tutorials/synth_sumb/wedge_130_60.mrc $PWD/data/tutorials/synth_sumb/rec/
fi
if [ -d $PWD/data/tutorials/synth_sumb/rec/particles ]; then
	rm -r $PWD/data/tutorials/synth_sumb/rec/particles/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/rec/particles
fi
if [ -d $PWD/data/tutorials/synth_sumb/rec/nali ]; then
	rm -r $PWD/data/tutorials/synth_sumb/rec/nali/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/rec/nali
fi
rm -r $PWD/data/tutorials/synth_sumb/class/*
rm -r $PWD/data/tutorials/synth_sumb/rln/*
if [ -d $PWD/data/tutorials/synth_sumb/org/ltomos ]; then
	rm -r $PWD/data/tutorials/synth_sumb/org/ltomos/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/org/ltomos
fi
if [ -d $PWD/data/tutorials/synth_sumb/org/uni_1st ]; then
	rm -r $PWD/data/tutorials/synth_sumb/org/uni_1st/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/org/uni_1st
fi
if [ -d $PWD/data/tutorials/synth_sumb/org/uni_2nd ]; then
	rm -r $PWD/data/tutorials/synth_sumb/org/uni_2nd/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/org/uni_2nd
fi
if [ -d $PWD/data/tutorials/synth_sumb/org/bi_2nd ]; then
	rm -r $PWD/data/tutorials/synth_sumb/org/bi_2nd/*
else
	mkdir -p $PWD/data/tutorials/synth_sumb/org/bi_2nd
fi
