"""

    Script for a classification in a reference (comparison is made by rlnImageName indexing)

    Input:  - The STAR files with the classification to compare
            - The STAR file with the ground truth particles

    Output: - An histogram with the fraction of particles per class in the reference classification STAR file

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyseg as ps

CLASS_COL, PART_COL = '_rlnClassNumber', '_rlnImageName'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/ribo_johannes'

# Input star files to compares
in_star_1 = ROOT_PATH + '/lum/klass_1/klass_1_gather.star'# '/klass_1/klass_1_gather.star' # '/lum_ext/stm/test3_ribo/klass_1_gather.star'
in_star_2 = ROOT_PATH + '/lum_ext/whole/class_nali/run1_c1_ct_it092_data_k2.star' # '/lum_ext/stm/test3_ribo/aln2_ribo.star'

# Getting subsets of particles
n_parts_low = 3000
n_parts_high = 3000
out_parts_low = ROOT_PATH + '/lum_ext/whole/class_nali/run1_c1_ct_it092_data_k2_low.star' # '/lum_ext/stm/test3_ribo/aln2_ribo_3000_low.star'
out_parts_high = ROOT_PATH + '/lum_ext/whole/class_nali/run1_c1_ct_it092_data_k2_high.star' # '/lum_ext/stm/test3_ribo/aln2_ribo_3000_high.star'

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

print 'Script for a classification in a reference.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput file with the classes: ' + in_star_1
print '\tInput file with reference classification: ' + in_star_2
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
imgs = star_2.get_column_data(PART_COL)

print '\tComputing particles in reference by class...'
nparts_lut, nref_lut = np.zeros(shape=len(classes_1), dtype=np.float32), np.zeros(shape=len(classes_1), dtype=np.float32)
parts_dic = dict()
for i in range(star_1.get_nrows()):
    hold_img, hold_class = star_1.get_element(PART_COL, i), star_1.get_element(CLASS_COL, i)
    nparts_lut[hold_class] += 1
    if hold_img in imgs:
        nref_lut[hold_class] += 1
    try:
        parts_dic[hold_class].append(i)
    except KeyError:
        parts_dic[hold_class] = list()
        parts_dic[hold_class].append(i)
recall = nref_lut / nparts_lut
sort_ids = np.argsort(recall)[::-1]

print '\tPrint results (Recall): '
klass_ids = np.arange(len(recall))
for idx in sort_ids:
    k_id = klass_ids[idx]
    print '\t\t-Klass ' + str(k_id) + ': ' + str(recall[k_id])

print '\tPlotting the histogram...'
plt.figure()
# plt.title('Classes recall')
plt.xlabel('AP Classes')
plt.ylabel('Recall')
plt.bar(klass_ids, recall[sort_ids])
# plt.xticks(klass_ids+0.5, sort_ids)
plt.show(block=True)

if out_parts_high is not None:
    print '\tSorting particles by class recall in: ' + out_parts_low + ', ' + out_parts_high
    shorted_parts = list()
    for idx in sort_ids:
        shorted_parts += parts_dic[idx]
    star_1_low, star_1_high = copy.deepcopy(star_1), copy.deepcopy(star_1)
    star_1_low.del_rows(shorted_parts[:len(shorted_parts)-n_parts_low])
    star_1_high.del_rows(shorted_parts[n_parts_high:])
    star_1_low.store(out_parts_low)
    star_1_high.store(out_parts_high)

print 'Terminated. (' + time.strftime("%c") + ')'