# command interpreter to be used
#$ -S /bin/bash
#
# name
#$ -N ref2_5gjv_rln
#
# merge stdout and stderr stream of job
#$ -j y
#
# output file
#$ -o ../../../../data/tutorials/synth_sumb/rln/5gjv/out_run1.log
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
#$ -pe openmpi 280
#
#
# set environment 
#mpirun echo $PATH
module load openmpi
module load RELION

# Relion refine settings
#!/bin/bash
printf -v relion_cmd "%s" `which relion_refine_mpi`
printf -v cmd "%s" "mpiexec -mca orte_forward_job_control 1 -n 21 --map-by ppr:3:node $relion_cmd --o ../../../../data/tutorials/synth_sumb/rln/5gjv/run1 --auto_refine --split_random_halves --i ../../../../data/tutorials/synth_sumb/rln/both_sides/run2_it025_data_c3.star --ref ../../../../data/tutorials/synth_sumb/rln/both_sides/run2_it025_class003.mrc --ini_high 60 --dont_combine_weights_via_disc --pool 3 --ctf --particle_diameter 320 --flatten_solvent --zero_mask --solvent_mask ../../../../data/tutorials/synth_sumb/rln/mask_cyl_130_30_110_40_r.mrc --oversampling 1 --healpix_order 2 --auto_local_healpix_order 3 --offset_range 5 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale  --j 1 --sigma_tilt 3.6667 --sigma_psi 3.6667 --angpix 2.62 --solvent_correct_fsc"
echo $cmd
$cmd




