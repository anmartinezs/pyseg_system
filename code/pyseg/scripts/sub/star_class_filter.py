"""

    Generates a filtered by class version from an input Relion's STAR file

    Input:  - STAR file
            - Classes to keep

    Output: - A copy of input STAR file where only entries of the specified classes are preserved

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyseg as ps

key_col, key_img = '_rlnClassNumber', '_rlnImageName'

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/psd_an/ex/syn/sub/relion/fils/pst_t/ch/pst_cont5'

# Input star file (file to be filtered)
in_star = ROOT_PATH + '/class_1/run15_it015_data.star'

# Classes star file (it can be the same as the filtered class)
in_star_c = ROOT_PATH + '/class_1/run15_it015_data.star'

# Class (or classes), minimum number of particles per group and output star file 3-Tuple.
# None can mean all classes or no regrouping
tuples = (((1,), None, ROOT_PATH+'/ref_2/c1/run15_it015_c1.star'),
         ((2,), None, ROOT_PATH+'/ref_2/c2/run15_it015_c2.star'),
         ((3,), None, ROOT_PATH+'/ref_2/c3/run15_it015_c3.star'),
         ((4,), None, ROOT_PATH+'/ref_2/c4/run15_it015_c4.star'),
         ((5,), None, ROOT_PATH+'/ref_2/c5/run15_it015_c5.star'),
         ((6,), None, ROOT_PATH+'/ref_2/c6/run15_it015_c6.star'),
         ((7,), None, ROOT_PATH+'/ref_2/c7/run15_it015_c7.star'),
         ((8,), None, ROOT_PATH+'/ref_2/c8/run15_it015_c8.star'),
         ((9,), None, ROOT_PATH+'/ref_2/c9/run15_it015_c9.star'),
         ((10,), None, ROOT_PATH+'/ref_2/c10/run15_it015_c10.star'))

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import os
import sys
import time
import copy

########## Print initial message

print('Filtering entries by classes in a STAR file.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tFile to split: ' + in_star)
print('\tFile with classes: ' + in_star_c)
print('\tClasses an grouping tuples: ')
for i, tup in enumerate(tuples):
    print('\t-Tuple ' + str(i) + ': ' + str(tup))
print('')

######### Process

print('Main Routine: ')

print('\tLoading input STAR files...')
hold_star = ps.sub.Star()
try:
    hold_star.load(in_star)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input STAR file could not be read because of "' + str(e.msg, e.expr) + '"')
    sys.exit(-1)
star_c = ps.sub.Star()
try:
    star_c.load(in_star_c)
except ps.pexceptions.PySegInputError as e:
    print('ERROR: input STAR file for classes could not be read because of "' + str(e.msg, e.expr) + '"')
    sys.exit(-1)

print('\tLoop for processing the input tuples...')
for i, tup in enumerate(tuples):
    star = copy.deepcopy(hold_star)
    print('\tProcessing tuple ' + str(i) + ':')
    classes = tup[0]
    if classes is None:
        classes = star_c.get_column_data(key_col)
    classes = set(classes)
    print('\t\t-Keeping classes in set: ' + str(classes))
    n_rows = star.get_nrows()
    del_rows = list()
    for i in range(star_c.get_nrows()):
        if not(star_c.get_element(key_col, i) in classes):
            img_name = star_c.get_element(key_img, i)
            idx_row = None
            for j in range(n_rows):
                try:
                    idx_row = star_c.find_element(key_img, img_name)
                except ValueError:
                    break
            if idx_row is not None:
                del_rows.append(idx_row)
    print('\t\t-Rows to delete: ' + str(len(del_rows)) + ' of ' + str(n_rows) + ' (survivors ' \
          + str(n_rows-len(del_rows)) + ')')
    print('\t\t-Deleting rows...')
    star.del_rows(del_rows)
    min_gp = tup[1]
    if min_gp is None:
        print('\t\t-Particle grouping, minimum group size: ' + str(min_gp))
        star.particle_group(min_gp)
        out_star = str(tup[2])
    print('\t\t-Storing the results in: ' + out_star)
    star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')