"""

    Computes statistics about class distribution by groups in a STAR file

    Input:  - STAR file

    Output: - Several figures for analyzing class distribution by gropus

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyseg as ps

GROUP_COL, CLASS_COL, MICRO_COL = '_rlnGroupNumber', '_rlnClassNumber', '_rlnMicrographName'
BAR_WIDTH = .35

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/in_situ_er'

# Input star file
in_star = ROOT_PATH + '/stat/rln/trans/class2_v3/run4cb_it025_data.star' # '/rln/dose_likes/moll/ref3_ip3r/run7c4_c4_it030_data.star' # '/rln/fatp_dmb_klass/3-dose_filt/ref1/run4_v2_k62_it025_data.star'

# Input parameters

lgd = True

# Grouping

# meta_groups = ( (1,), (2,), (3,), (4,))
# meta_groups = ( (10,), (11,), (12,), (13,), (14,), (15,), (16,), (26,), (27,))
# meta_groups = ( (7,), (8,), (9,), (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,))
# (Uli-Ctrl, Uli-PDBu, Ctrl, Stim)
# meta_groups = ((1, 2, 3, 4),
#                (5, 6),
#                (10, 11, 12, 13, 14, 15, 16, 26, 27),
#                (7, 8, 9, 17, 18, 19, 20, 21, 22, 23, 24, 25)) # None
meta_groups = ((0,4,12,15),
               (1, 2),
               (3,4,6,9,10,11,14,18),
               (7,8,13,17,19,20,21,22))

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

print 'Statistical analysis of class distribution by group from a STAR file.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput file: ' + in_star
if meta_groups is None:
    print '\tNo meta groups.'
else:
    print '\tMeta groups: ' + str(meta_groups)
if lgd:
    print '\tPrint legends'
print ''

######### Process

print 'Main Routine: '

print '\tLoading input STAR file...'
star = ps.sub.Star()
try:
    star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"'
    sys.exit(-1)

print '\tFinding the set of groups...'
groups = star.get_column_data_set(GROUP_COL)
if groups is None:
    print '\t-WARNING: No tomograms found!.'
    print 'Terminated. (' + time.strftime("%c") + ')'
print '\t-Number of tomograms found: ' + str(len(groups))
groups = list(np.sort(np.asarray(list(groups), dtype=np.int)))

print '\tFinding micrographs name per group...'
for group in groups:
    hold_star = copy.deepcopy(star)
    del_rows = list()
    for i in range(hold_star.get_nrows()):
        if hold_star.get_element(GROUP_COL, i) != group:
            del_rows.append(i)
    hold_star.del_rows(del_rows)
    micros = hold_star.get_column_data_set(MICRO_COL)
    print '\t- Group ' + str(group) + ': ' + str(micros)

print '\tFinding the set of classes...'
classes = star.get_column_data_set(CLASS_COL)
if classes is None:
    print '\t-WARNING: No class found!.'
    print 'Terminated. (' + time.strftime("%c") + ')'
print '\t-Number of classes found: ' + str(len(classes))
classes = list(classes)

if meta_groups is None:
    print '\tComputing particles by group...'
    groups_str, vals_g = list(), list()
    for group in groups:
        val = star.count_elements(((GROUP_COL, group),))
        if val is not None:
            vals_g.append(float(val))
        else:
            vals_g.append(0.)
        groups_str.append(str(group))
else:
    print '\tComputing particles by meta-group...'
    groups_str, vals_g = list(), list()
    for i, meta_group in enumerate(meta_groups):
        hold_val = 0.
        for group in meta_group:
            val = star.count_elements(((GROUP_COL, group),))
            if val is not None:
                hold_val += float(val)
        vals_g.append(hold_val)
        groups_str.append(str(i))
# fig, ax = plt.subplots()
fig = plt.figure(1)
plt.ylabel('Number of particles')
if meta_groups is None:
    plt.xlabel('Group')
    plt.title('Particles by group')
    index = np.arange(len(groups))
else:
    plt.xlabel('Meta-Group')
    plt.title('Particles by meta-group')
    index = np.arange(len(meta_groups))
plt.bar(index+.5*BAR_WIDTH, vals_g, BAR_WIDTH)
plt.xticks(index + BAR_WIDTH, groups_str)
plt.tight_layout()
plt.show(block=False)

print '\tComputing particles by class...'
classes_str, vals = list(), list()
for klass in classes:
    val = star.count_elements(((CLASS_COL, klass),))
    if val is not None:
        vals.append(float(val))
    else:
        vals.append(0.)
    classes_str.append(str(klass))
# fig, ax = plt.subplots()
fig = plt.figure(2)
index = np.arange(len(classes))
plt.bar(index+.5*BAR_WIDTH, vals, BAR_WIDTH)
plt.xlabel('Class')
plt.ylabel('Number of particles')
plt.title('Particles by class')
plt.xticks(index + BAR_WIDTH, classes_str)
plt.tight_layout()
plt.show(block=False)

if meta_groups is None:
    print '\tComputing class proportions by group...'
    groups_p = np.zeros(shape=(len(groups), len(classes)), dtype=np.float)
    for i in range(len(groups)):
        if vals_g[i] > 0:
            for j in range(len(classes)):
                val = star.count_elements(((GROUP_COL, groups[i]), (CLASS_COL, classes[j]),))
                if val is not None:
                    groups_p[i][j] = float(val) / vals_g[i]
else:
    print '\tComputing class proportions by meta-group...'
    groups_p = np.zeros(shape=(len(meta_groups), len(classes)), dtype=np.float)
    for i in range(len(meta_groups)):
        if vals_g[i] > 0:
            for j in range(len(classes)):
                hold_val = 0.
                for k in range(len(meta_groups[i])):
                    hold_val += star.count_elements(((GROUP_COL, meta_groups[i][k]), (CLASS_COL, classes[j]),))
                groups_p[i][j] = float(hold_val) / vals_g[i]
# fig = plt.figure(3)
fig, ax = plt.subplots()
plt.ylabel('Prop. of particles')
if meta_groups is None:
    plt.xlabel('Group')
    plt.title('Proportion of classes by group')
    offset = np.zeros(shape=len(groups), dtype=np.float)
else:
    plt.xlabel('Meta Group')
    plt.title('Proportion of classes by meta-group')
    offset = np.zeros(shape=len(meta_groups), dtype=np.float)
index = np.arange(len(groups_p))
colors = cm.rainbow(np.linspace(0, 1, len(classes)))
for j in range(len(classes)):
    ax.bar(index+.5*BAR_WIDTH, groups_p[:, j], BAR_WIDTH, color=colors[j], bottom=offset, label=classes_str[j])
    offset += groups_p[:, j]
plt.xticks(index + BAR_WIDTH, groups_str)
if lgd:
    plt.legend()
plt.tight_layout()
plt.show(block=True)

print 'Terminated. (' + time.strftime("%c") + ')'