# command interpreter to be used
#$ -S /bin/bash
#
# name
#$ -N mb_align_all
#
# merge stdout and stderr stream of job
#$ -j y
#
# output file
#$ -o ../../../../data/tutorials/synth_sumb/rec/nali/out_run1.log
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
# source ~/.bashrc.modules
# module unload RELION/1.4
# module unload RELION/2.03
module load RELION

# mpiexec --tag-output -n $NSLOTS `which relion_refine_mpi`
mpiexec -mca orte_forward_job_control 1 -n 21 --map-by ppr:3:node `which relion_refine_mpi` --o ../../../../data/tutorials/synth_sumb/rec/nali/run1 --auto_refine --split_random_halves --i ../../../../data/tutorials/synth_sumb/rec/particles_rln.star --ref ../../../../data/tutorials/synth_sumb/rec/dr_particles.mrc --ini_high 60 --dont_combine_weights_via_disc --pool 3 --ctf --particle_diameter 320 --flatten_solvent --zero_mask --solvent_mask ../../../../data/tutorials/synth_sumb/rec/mask_sph_130_60_r.mrc --solvent_correct_fsc  --oversampling 1 --healpix_order 2 --auto_local_healpix_order 3 --offset_range 5 --offset_step 2 --sym C10 --low_resol_join_halves 40 --norm --scale  --j 1 --angpix 2.62 --firstiter_cc --sigma_tilt 3.667 --sigma_psi 3.667


