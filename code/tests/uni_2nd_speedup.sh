# command interpreter to be used
#$ -S /bin/bash
#
# name
#$ -N uni_2nd_analysis
#
# merge stdout and stderr stream of job
#$ -j y
#
# output file
#$ -o ./results/uni_2nd_speedup.log
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
#$ -pe openmpi 40


# set environment
# module unload PYTHON/3.4.0
# module load PYTHON/2.7.9
PYTHONPATH="${PYTHONPATH}:$PWD/../"

#
printf -v cmd "%s" "python ../pyorg/scripts/tests/uni_2nd_speedup.py $PWD"
echo $cmd
$cmd

