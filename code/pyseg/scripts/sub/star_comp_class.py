"""

    Script for comparing two classifications

    Input:  - The STAR files with the classification to compare

    Output: - Table with the similitude (number of shared particle / number of particles per class) among classes

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyseg as ps

CLASS_COL, PART_COL = '_rlnClassNumber', '_rlnImageName'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst_t/ch'

# Input star files to compares
in_star_1 = ROOT_PATH + '/pst_cont5/class_batch/run1_c8_5_it030_data.star'
in_star_2 = ROOT_PATH + '/pst_cont5/class_batch/run1_c8_15_it030_data.star'

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import sys
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

########## Print initial message

print 'Similitude comparison between two classifications.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput file 1: ' + in_star_1
print '\tInput file 2: ' + in_star_2
print ''

######### Process

print 'Main Routine: '

print '\tLoading input STAR file 1: ' + in_star_1
star_1 = ps.sub.Star()
try:
    star_1.load(in_star_1)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"'
    sys.exit(-1)
classes_1 = list(set(star_1.get_column_data(CLASS_COL)))
print '\t\t-' + str(len(classes_1)) + ' classes found: ' + str(classes_1)

print '\tLoading input STAR file 2: ' + in_star_2
star_2 = ps.sub.Star()
try:
    star_2.load(in_star_2)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"'
    sys.exit(-1)
classes_2 = list(set(star_2.get_column_data(CLASS_COL)))
print '\t\t-' + str(len(classes_1)) + ' classes found: ' + str(classes_1)

print '\tComputing similitude matrix (1 to 2)...'
nparts_1 = np.zeros(shape=max(classes_1)+1, dtype=np.int)
sim_mat_1 = np.zeros(shape=(max(classes_1)+1,max(classes_2)+1), dtype=np.float)
parts_1, pclasses_1 = star_1.get_column_data(PART_COL), star_1.get_column_data(CLASS_COL)
parts_2, pclasses_2 = star_2.get_column_data(PART_COL), star_2.get_column_data(CLASS_COL)
for part_1, pclass_1 in zip(parts_1, pclasses_1):
    nparts_1[pclass_1] += 1
    try:
        idx = parts_2.index(part_1)
        pclass_2 = pclasses_2[idx]
        sim_mat_1[pclass_1, pclass_2] += 1
    except ValueError:
        pass
for i in range(1, sim_mat_1.shape[0]):
    sim_mat_1[i, :] /= float(nparts_1[i])

print '\tComputing similitude matrix (2 to 1)...'
nparts_2 = np.zeros(shape=max(classes_2)+1, dtype=np.int)
sim_mat_2 = np.zeros(shape=(max(classes_2)+1,max(classes_1)+1), dtype=np.float)
for part_2, pclass_2 in zip(parts_2, pclasses_2):
    nparts_2[pclass_2] += 1
    try:
        idx = parts_1.index(part_2)
        pclass_1 = pclasses_1[idx]
        sim_mat_2[pclass_2, pclass_1] += 1
    except ValueError:
        pass
for i in range(1, sim_mat_2.shape[0]):
    sim_mat_2[i, :] /= float(nparts_2[i])

print '\tSIMILITUDE FOR 1 TO 2 CASE:'
for i in range(1, sim_mat_1.shape[0]):
    print '\t\t-CLASS ' + str(classes_1[i-1]) + ': ' + str(sim_mat_1[i, 1:])
    print '\t\t\t+Max: ' + str(sim_mat_1[i, 1:].max())
    print '\t\t\t+Min: ' + str(sim_mat_1[i, 1:].min())
    print '\t\t\t+Mean: ' + str(sim_mat_1[i, 1:].mean())
    print '\t\t\t+Std: ' + str(sim_mat_1[i, 1:].std())
    print '\t\t\t+Sum: ' + str(int(sim_mat_1[i, 1:].sum()) * nparts_1[i])

print '\tSIMILITUDE FOR 2 TO 1 CASE:'
for i in range(1, sim_mat_2.shape[0]):
    print '\t\t-CLASS ' + str(classes_2[i-1]) + ': ' + str(sim_mat_2[i, 1:])
    print '\t\t\t+Max: ' + str(sim_mat_2[i, 1:].max())
    print '\t\t\t+Min: ' + str(sim_mat_2[i, 1:].min())
    print '\t\t\t+Mean: ' + str(sim_mat_2[i, 1:].mean())
    print '\t\t\t+Std: ' + str(sim_mat_2[i, 1:].std())
    print '\t\t\t+Sum: ' + str(int(sim_mat_2[i, 1:].sum()) * nparts_2[i])

print 'Terminated. (' + time.strftime("%c") + ')'