#!/usr/bin/bash -l
#SBATCH -D .
# SBATCH -o pyseg_all-%j.out
# SBATCH -e pyseg_all-%j.err
#SBATCH --mem 256G
#SBATCH -J pyseg
#SBATCH -N 1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=medium
#SBATCH --time=48:00:00

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

# Inputs
PIXELSIZE="0.756"
TOMOSTAR="bin4_1_seg.star"
PICKXML="../pick/in/mb_cont_1.xml"

# Output directories
OUTDIR="Pipeline"
GRAPHSDIR="$OUTDIR/graphs"
FILDIR="$OUTDIR/fils"
PICKDIR="$OUTDIR/pick"

## END BATCH HEADER ##

mkdir -pv $OUTDIR

rm -rv $GRAPHSDIR 2> /dev/null
python3 mb_graph_1p.py --inStar $TOMOSTAR --outDir $GRAPHSDIR --pixelSize $PIXELSIZE

rm -rv $FILDIR 2> /dev/null
GRAPHSTAR="${GRAPHSDIR}/$(basename ${TOMOSTAR%.star}_mb_graph.star)"
if [[ -e $GRAPHSTAR ]] ; then
  python3 mb_fils_network.py --inStar $GRAPHSTAR --outDir $FILDIR
else
  echo "ERROR!! Graphs output '$GRAPHSTAR' not found! Exiting..."
  exit
fi

rm -rv $PICKDIR 2> /dev/null
TARGETSTAR="$FILDIR/fil_mb_sources_to_no_mb_targets_net.star"  # currently hardwired based on mb_fils_network.py
if [[ -e $TARGETSTAR ]] ; then
  python3 mbu_picking.py --inStar $TARGETSTAR --inSlices $PICKXML --outDir $PICKDIR
else
  echo "ERROR!! Fils output '$TARGETSTAR' not found! Exiting..."
  exit
fi

