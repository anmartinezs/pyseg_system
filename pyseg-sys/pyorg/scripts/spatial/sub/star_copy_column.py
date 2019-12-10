"""

    Copies column(s) values from a STAR file to another STAR file, if targe file already contained a set column old values
    are overwritten

    Input:  - Source STAR file
            - Target STAR file
            - List with te column keys to copy

    Output: - A copy of target STAR file with column values indicated set from source STAR file

"""

__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

################# Package import

import os
import sys
import time
import pyto
import pyseg as ps
import numpy as np

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = ''

# Source STAR
in_star_source = ROOT_PATH + ''

# Target STAR
in_star_target = ROOT_PATH + ''

# Output STAR file
out_star = ROOT_PATH + ''

# Path to store output subvolumes with noise
keys = ('_rlnImageName')

########################################################################################
# Global functions
########################################################################################

########################################################################################
# MAIN ROUTINE
########################################################################################

########## Print initial message

print 'Create a mirrored particles STAR with noise.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tSource STAR file: ' + str(in_star_source)
print '\tTarget STAR file: ' + str(in_star_target)
print '\tOutput STAR file: ' + str(out_star)
print '\tColumn keys to copy: ' + str(keys)
print ''

######### Process

print 'Main Routine: '

print '\tLoading input STAR files...'
star_source, star_target = ps.sub.Star(), ps.sub.Star()
try:
    star_source.load(in_star_source)
    star_target.load(in_star_target)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input list of STAR files could not be loaded because of "' + e.get_message() + '"'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
if star_source.get_nrows() != star_target.get_nrows():
    print 'ERROR: number of rows for input STAR files do not agree!'
    print 'Terminated. (' + time.strftime("%c") + ')'
    sys.exit(-1)
for key in star_source.get_column_keys():
    if key not in keys:
        print 'ERROR: STAR file ' + str(in_star_source) + ' do no contain column ' + str(key)
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
for key in star_target.get_column_keys():
    if key not in keys:
        print 'ERROR: STAR file ' + str(in_star_target) + ' do no contain column ' + str(key)
        print 'Terminated. (' + time.strftime("%c") + ')'
        sys.exit(-1)
star_out = np.copy(star_target)

print '\tCoping column values...'
for key in keys:
    star_out.set_column_data(key, star_source.get_column_data(key))

print 'Storing output STAR file in: ' + out_star
star_out.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'