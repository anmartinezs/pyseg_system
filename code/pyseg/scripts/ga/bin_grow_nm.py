"""

    Script for growing a binary segmentation the specified nm

    Input:  - Segmentation
            - Distance to grow

    Output: - Grown segmentation

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
import operator
import scipy as sp
import pyseg as ps
import numpy as np

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomo> -o <output_tomo> -d <float> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <input_tomo>: Input tomogram with the segmentation (treated as binary, bg=0 and fg>0). \n' + \
           '    -o <output_tomo>: Output binary segmentation.' + \
           '    -d <float>(optional): distance to grow in nm' + \
           '    -r <float>(optional): tomogram resolution in nm/voxel (default 1.0).' + \
           '    -v (optional): verbose mode activated (default False).'

################# Work routine

def do_bin_grow_nm(in_tomo, out_tomo, dst, res, verbose):

    # Initialization
    if verbose:
        print '\tLoading input tomogram...'
    tomo = ps.disperse_io.load_tomo(in_tomo)


    if verbose:
        print '\tDistance transform...'
    tomo_dst = sp.ndimage.morphology.distance_transform_edt(tomo==0) * res

    if verbose:
        print '\tThresholding...'
    tomo_dst = tomo_dst <= dst

    if verbose:
        print '\tStoring the result...'
    ps.disperse_io.save_numpy(tomo_dst, out_tomo)

################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvi:o:d:r:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    in_tomo = None
    out_tomo = None
    dst = None
    res = 1.
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print usage_msg
            print help_msg
            sys.exit()
        elif opt == "-i":
            in_tomo = str(arg)
        elif opt == "-o":
            out_tomo = str(arg)
        elif opt == "-d":
            dst = float(arg)
        elif opt == "-r":
            res = float(arg)
        elif opt == "-v":
            verbose = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(3)

    if (in_tomo is None) or (out_tomo is None) or (dst is None):
        print usage_msg
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print 'Running tool for growing a binary segmentation.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + in_tomo
            print '\tOutput file: ' + out_tomo
            print '\tResolution: ' + str(res) + ' nm/voxel'
            print '\tDistance: ' + str(dst) + ' nm'
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_bin_grow_nm(in_tomo, out_tomo, dst, res, verbose)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'

if __name__ == "__main__":
    main(sys.argv[1:])