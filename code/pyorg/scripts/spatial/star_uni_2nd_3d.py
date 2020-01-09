"""

    Script for doing Univariate statistical 2nd order spatial analysis for a 3D distributed point pattern

    Input:  - Star file with surfaces list
            - Parameters for setting the statistical analysis

    Output: - A numpy Matrix for computing the 2nd order analysis for every particle

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import gc
import pyorg as sd
import os
import sys
import math
import time
import numpy as np

########################################################################################
# PARAMETERS
########################################################################################

# INPUT
ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst/test/surf/out'
in_pkl = ROOT_PATH + '/ref_8_m10_k6_tpl.pkl'

# OUTPUT

out_stem = 'test_uni_2nd'
out_dir = ROOT_PATH + '/stat_mat'

# NHOOD

nh_rad_rg = np.arange(2, 80, 3) # nm
nh_thick = None
nh_border = True
nh_res_arc = 10

# SIMULATION

sim_model = None # sd.surf.ModelCSRV # If None disabled
sim_nins = 1000
sim_part_surf = ROOT_PATH + '/ref_8_m10_k6_surf_ref.vtp'
sim_emb_mode = 'center'

# COMPUTATION

cp_npr = 2 # None means Auto

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message

print 'Univariate 2nd order spatial 3D distribution.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput list of tomograms with surface particles: ' + str(in_pkl)
print '\tOutput directory: ' + out_dir
print '\tNeighborhood properties:'
print '\t\t-Radius array: ' + str(nh_rad_rg) + ' nm'
if nh_thick is None:
    print '\t\t-Shape: sphere'
else:
    print '\t\t-Shell with thickness: ' + str(nh_thick) + ' nm'
print '\t\t-Arc resolution (number samples circle): ' + str(nh_res_arc)
if nh_border:
    print '\t\t-Border compesation activated.'
else:
    print '\t\t-NO border compensation.'
if sim_model is not None:
    print '\tRandom model simulation:'
    print '\t\t-Simulator: ' + str(sim_model)
    print '\t\t-Number of instances: ' + str(sim_nins)
    print '\t\t-Particle surface: ' + str(sim_part_surf)
    print '\t\t-VOI embedding mode: ' + str(sim_emb_mode)
print '\tComputation:'
if cp_npr is None:
    print '\t\t-Number of processes: Auto'
else:
    print '\t\t-Number of processes: ' + str(cp_npr)
print ''

# Process

print 'Main Routine: '

print '\tLoading input data...'
ltomos = sd.globals.unpickle_obj(in_pkl)

if sim_model is not None:
    mat = np.zeros(shape=(ltomos.get_num_particles(), len(nh_rad_rg)), dtype=np.float)
    for i in range(sim_nins):
        htomos = ltomos.gen_model_instance(sim_model, sim_part_surf, mode=sim_emb_mode)
        mat += ltomos.compute_uni_2nd_order(nh_rad_rg, nh_res_arc, thick=nh_thick, border=nh_border,
                                            npr=cp_npr)
    mat /= float(sim_nins)
else:
    print '\tComputing matrix from experimental data...'
    mat = ltomos.compute_uni_2nd_order(nh_rad_rg, nh_res_arc, thick=nh_thick, border=nh_border,
                                       npr=cp_npr)


out_mat = out_dir + '/' + out_stem + '_uni_2nd_mat.npy'
print '\tStoring computed matrix in:'
mat.save(out_mat)

print 'Terminated. (' + time.strftime("%c") + ')'
