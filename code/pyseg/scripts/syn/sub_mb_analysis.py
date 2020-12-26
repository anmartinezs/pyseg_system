"""

    Script for analyzing Z density profiles from subvolumes with aligned membrane normal

    Input:  - The path to a subvolume

    Output: - A density profile around Z-axis is computed

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import os
import gc
import sys
import time
import copy
import random
import pyseg as ps
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt, rcParams

from pyseg import sub, pexceptions
from pyseg.sub.variables import RadialAvg3D

########## Global variables

__author__ = 'Antonio Martinez-Sanchez'

BAR_WIDTH = .35

rcParams['axes.labelsize'] = 22 # 14
rcParams['xtick.labelsize'] = 22 # 14
rcParams['ytick.labelsize'] = 22 # 14

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/pool/pool-lucic2'

# Input STAR file
in_svol = ROOT_PATH + '/christos/workspace/glur/rln/pst/ref_3_nocont/run4_l2_class001.mrc' # '/antonio/workspace/psd_an/ex/syn3/rln/pst/ref2_model_subclean/run4_h0_class001.mrc' #

####### Output data

out_graph = None # ROOT_PATH + '/christos/workspace/glur/rln/pst/figs/density_profiles/ampar_de_novo_old.png'

####### Particles pre-processing settings



####### Multiprocessing settings



########################################################################################
# Local functions
########################################################################################



########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print('Compute density profile and Z-aligned sub-volumes.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput sub-volume file: ' + in_svol)
if out_graph is not None:
    print('\tOutput file for the density profile: ' + out_graph)
print('')


print('Loading input sub-volume file...')
try:
    svol = ps.disperse_io.load_tomo(in_svol, mmap=True)
except pexceptions.PySegInputError as e:
    print('ERROR: input sub-volume file could not be loaded because of "' + e.get_message() + '"')
    print('Terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)

print('Computing the rotational average...')
averager = RadialAvg3D(svol.shape, axis='z')
rsvol = averager.avg_vol(svol)
lsvol = np.sum(rsvol, axis=1)
lsvol /= lsvol.sum()

print('Plotting density profile...')
plt.figure()
plt.title('Density profile around Z-axis')
plt.ylabel('Normalized density')
plt.xlabel('Z-axis [pixels]')
plt.plot(range(len(lsvol)), lsvol, color='b', linestyle='-')
plt.legend(loc=4)
plt.tight_layout()
if out_graph is None:
    plt.show(block=True)
else:
    plt.savefig(out_graph)
plt.close()


print('Successfully terminated. (' + time.strftime("%c") + ')')

