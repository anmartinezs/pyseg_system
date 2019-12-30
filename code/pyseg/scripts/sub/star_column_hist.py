"""

    Display's a histogram to analyze the statistical distribution of a column data in a STAR file, or a combination
    of them.

    Input:  - STAR file(s)/column(s)
            - Operator (optional)

    Output: - A figures with the histogram is displayed

"""


__author__ = 'Antonio Martinez-Sanchez'


###### Global variables

import pyto
import pyseg as ps
import operator

########## Useful operators

# Angular difference in degrees
def op_ang_dif_degs(ang_1, ang_2):
    angr_1, angr_2 = np.radians(ang_1), np.radians(ang_2)
    return np.degrees(np.arccos(np.cos(angr_1)*np.cos(angr_2)+np.sin(angr_1)*np.sin(angr_2)))

def op_ang_eu_degs(eu_1, eu_2):
    eur_1, eur_2 = np.radians(eu_1), np.radians(eu_2)
    return np.degrees(pyto.geometry.Rigid3D.angle_between_eulers(eur_1, eur_2, mode='zyz_in_active'))

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/ribo_johannes/lum_ext_repick/ribo'

# Input STAR file(s)
in_star_1 = ROOT_PATH + '/ref_rln/run2_rt2s_ssup_data.star' # '/clean/ref_1/recon/clean_ref_run1_data_malign.star' # ROOT_PATH + '/particles_pst_cont_prior_rnd_rot.star'
in_star_2 = None # ROOT_PATH + '/nali/run1_c10_it017_data.star' # ROOT_PATH + '/class_1/run14_it035_data.star' # It can be None

# Input column(s) in the STAR files, if '_rlnRotations' the three columns for rotation angles are considered
# an the operator is set to 'op_ang_eu_degs'
in_col_1 = '_rlnCoordinateX' # '_rlnMaxValueProbDistribution'
in_col_2 = '_rlnAngleRot' # '_rlnOriginY' # '_rlnAnglePsi' # It can be None

# Operator function if between columns
in_op = op_ang_dif_degs # None

###### Histogram parameters

hist_nbins = 40
hist_norm = False
hist_rg = (0, 180) # (0., 1.)

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import time
import sys
import numpy as np
import matplotlib.pyplot as plt

########## Print initial message

print 'Histogram analysis a STAR file(s).'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tInput file 1: ' + in_star_1
if in_col_1 == '_rlnRotation':
    print '\tAnalyzing distance rotations'
    if in_star_2 is not None:
        print '\tInput file 2: ' + in_star_2
else:
    print '\t\t-Input column: ' + in_col_1
    if in_star_2 is not None:
        print '\tInput file 2: ' + in_star_2
        print '\t\t-Input column: ' + in_col_2
        print '\t\t-Operator: ' + str(in_op)
print '\tHistogram parameters: '
print '\t\t-Number of bins: ' + str(hist_nbins)
if hist_norm:
    print '\t\t-Normalized'
print ''

######### Process

print 'Main Routine: '

print '\tLoading input STAR file(s)...'
star_1 = ps.sub.Star()
try:
    star_1.load(in_star_1)
except ps.pexceptions.PySegInputError as e:
    print 'ERROR: input STAR file \'' + str(in_star_1) +  '\' could not be read because of "' + str(e.msg) + ', ' + str(e.expr) + '"'
    sys.exit(-1)
if in_star_2 is not None:
    star_2 = ps.sub.Star()
    try:
        star_2.load(in_star_2)
    except ps.pexceptions.PySegInputError as e:
        print 'ERROR: input STAR file \'' + str(in_star_2) +  '\' could not be read because of "' + str(e.msg) + ', ' + str(e.expr) + '"'
        sys.exit(-1)

print '\tLoading column(s) data...'
if in_col_1 == '_rlnRotation':
    if in_star_2 is None:
        rots, tilts, psis = star_1.get_column_data('_rlnAngleRot'), star_1.get_column_data('_rlnAngleTilt'), \
                            star_1.get_column_data('_rlnAnglePsi')
        if (rots is None) or (tilts is None) or (psis is None):
            print 'ERROR: one or more of the rotation angles are not present in the STAR file ' + in_star_1
            sys.exit(-1)
        print '\t\t-Performing data column operation...'
        dat = list()
        rot_ref = np.zeros(shape=3, dtype=np.float)
        for i in range(star_1.get_nrows()):
            dat = op_ang_eu_degs(np.asarray((rots[i], tilts[i], psis[i]), dtype=np.float), rot_ref)
    else:
        dat = list()
        dat_2 = star_2.get_column_data(in_col_2)
        if dat is None:
            print 'ERROR: column \'' + str(in_col_2) + '\' is not present in STAR file \'' + str(in_star_2) + '\''
            sys.exit(-1)
        print '\t\t-Performing data column operation (pairing)...'
        for row in range(star_1.get_nrows()):
            mic = star_1.get_element('_rlnImageName', row)
            try:
                row_2 = star_2.find_element('_rlnImageName', mic)
            except ValueError:
                continue
            rot1, tilt1, psi1 = star_1.get_element('_rlnAngleRot', row), star_1.get_element('_rlnAngleTilt', row), \
                                star_1.get_element('_rlnAnglePsi', row)
            if (rot1 is None) or (tilt1 is None) or (psi1 is None):
                print 'ERROR: one or more of the rotation angles are not present in the STAR file ' + in_star_1
                sys.exit(-1)
            rot2, tilt2, psi2 = star_2.get_element('_rlnAngleRot', row_2), star_2.get_element('_rlnAngleTilt', row_2), \
                                star_2.get_element('_rlnAnglePsi', row_2)
            if (rot2 is None) or (tilt2 is None) or (psi2 is None):
                print 'ERROR: one or more of the rotation angles are not present in the STAR file ' + in_star_2
                sys.exit(-1)
            dat.append(op_ang_eu_degs(np.asarray((rot1, tilt1, psi1), dtype=np.float),
                                      np.asarray((rot2, tilt2, psi2), dtype=np.float)))
else:
    if in_star_2 is None:
        dat = star_1.get_column_data(in_col_1)
        if dat is None:
            print 'ERROR: column \'' + str(in_col_1) + '\' is not present in STAR file \'' + str(in_star_1) + '\''
            sys.exit(-1)
    else:
        dat = list()
        dat_2 = star_2.get_column_data(in_col_2)
        if dat is None:
            print 'ERROR: column \'' + str(in_col_2) + '\' is not present in STAR file \'' + str(in_star_2) + '\''
            sys.exit(-1)
        print '\t\t-Performing data column operation (pairing)...'
        for row in range(star_1.get_nrows()):
            mic = star_1.get_element('_rlnImageName', row)
            try:
                row_2 = star_2.find_element('_rlnImageName', mic)
            except ValueError:
                continue
            dat_1, dat_2 = star_1.get_element(in_col_1, row), star_2.get_element(in_col_2, row_2)
            dat.append(in_op(dat_1, dat_2))

if len(dat) == 0:
    print 'WARNING: Nothing to plot'
else:
    print '\tPlotting the histogram...'
    dat = np.asarray(dat, dtype=np.float)
    plt.hist(dat, bins=hist_nbins, range=hist_rg, normed=hist_norm)
    plt.show(block=True)
    plt.close()

print 'Terminated. (' + time.strftime("%c") + ')'