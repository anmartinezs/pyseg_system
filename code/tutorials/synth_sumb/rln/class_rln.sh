# command interpreter to be used
#$ -S /bin/bash
#
# name
#$ -N class_rln
#
# merge stdout and stderr stream of job
#$ -j y
#
# output file
#$ -o ../../../../data/tutorials/synth_sumb/rln/medium_1/out_run4.log
#
# use current working directory
#$ -cwd
#
# define mail notification events
#$ -m n
#
# request the given resources
# maximal run time 7 days
#$ -l h_rt=500000
#
# request the given resources
# maximal memory for the job 192G or 512G
# -l h_vmem=192G
#
# request slot range for parallel jobs
#$ -pe openmpi 80
#
#
# set environment
#mpirun echo $PATH
module load openmpi
# source ~/.bashrc.modules
module load RELION

# Relion refine settings
#!/bin/bash
printf -v relion_cmd "%s" `which relion_refine_mpi`
printf -v cmd "%s" "mpiexec --tag-output -n $NSLOTS $relion_cmd --o ../../../../data/tutorials/synth_sumb/rln/medium_1/run4 --i ../../../../data/tutorials/synth_sumb/rln/medium_1.star --ref ../../../../data/tutorials/synth_sumb/rln/mbf_align_all/run1_ref_k7_class001.mrc --ini_high 60 --dont_combine_weights_via_disc --pool 3 --ctf --particle_diameter 320 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 3 --offset_range 5 --offset_step 2 --sym C1 --norm --scale  --j 1 --angpix 2.62 --sigma_tilt 3.6667 --sigma_psi 3.6667  --solvent_correct_fsc --K 3 --iter 25 --solvent_mask ../../../../data/tutorials/synth_sumb/rln/mask_cyl_130_35_100_40_r.mrc --tau2_fudge 4 --firstiter_cc"
echo $cmd
$cmd


