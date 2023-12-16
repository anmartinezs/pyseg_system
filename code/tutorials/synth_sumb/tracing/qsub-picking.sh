#!/usr/bin/bash -l
#SBATCH -D .
# SBATCH -o pyseg_picking-%j.out
# SBATCH -e pyseg_picking-%j.err
#SBATCH --mem 24G
#SBATCH -J pyseg
#SBATCH -N 1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=medium
#SBATCH --time=10:00

# SLURM_NTASKS will be undefined if number of nodes and tasks-per-node are specified
if [[ -z $SLURM_NTASKS ]] ; then
  SLURM_NTASKS=$(($SLURM_JOB_NUM_NODES * $SLURM_TASKS_PER_NODE))

  # Sanity check
  if [[ -z $SLURM_NTASKS ]] ; then
    echo "ERROR!! SLURM_NTASKS is undefined! Exiting..."
    exit
  fi
fi

# Diagnostics
echo "SLURM_JOB_NUM_NODES : '$SLURM_JOB_NUM_NODES'"
echo "SLURM_TASKS_PER_NODE : '$SLURM_TASKS_PER_NODE'"
echo "SLURM_NTASKS : '$SLURM_NTASKS'"
#echo " : '$'"
echo

module purge
shopt -s expand_aliases
source /usr/users/rubsak/sw/rubsak.bashrc
use_pyseg

python3 mbu_picking.py --inStar ../fils/out/fil_mb_sources_to_no_mb_targets_net.star --inSlices ../pick/in/mb_cont_1.xml --outDir ../pick/out

