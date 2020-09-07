"""

    Contrast Limited Adapted Histogram Equalization (CLAHE) for a GraphMCF property
    Adaptation to GraphMCF of:
    A. M. Rez, Journal of VLSI Signal Processing 30, 35-44, 2004

    Input:  - GraphMCF (pkl)
            - Property to equalize
            - CLAHE setting

    Output: - Equalized property added to the GraphMCF

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import os
import sys
import time
import getopt
import pyseg as ps
from factory import unpickle_obj
try:
    import pickle as pickle
except:
    import pickle

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_pkl> -p <prop_key> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <string>: path to input GraphMCF pickle file. \n' + \
           '    -p <string>: property key to equalize. \n' + \
           '    -n <int>(optional): number of neighbours and greyscales (default 4). \n' + \
           '    -c <float>(optional): clipping factor percentage (default 100%). \n' + \
           '    -s <float>(optional): maximum slope (default 4). \n' + \
           '    -v (optional): verbose mode activated.'

################# Helping routines

################# Work routine

def do_clahe(input_pkl, prop_key, N, clip_f, s_max, verbose):

    # Initialization
    if verbose:
        print('\tLoading graph...')
    path, stem = os.path.split(input_pkl)
    stem, _ = os.path.splitext(stem)
    graph = unpickle_obj(input_pkl)

    if verbose:
        print('\tApplying CLAHE...')
    graph.clahe_prop(prop_key, N, clip_f, s_max)

    if verbose:
        print('\tStoring the result...')
    graph.pickle(input_pkl)
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            path + '/' + stem + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                            path + '/' + stem + '_sch.vtp')
    if verbose:
        print('\tFile ' + input_pkl + ' overwritten.')


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvi:p:n:c:s")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_pkl = None
    prop_key = None
    N = 256
    clip_f = 100.
    s_max = 4.
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_pkl = arg
        elif opt == "-p":
            prop_key = arg
        elif opt == "-n":
            N = int(arg)
        elif opt == "-c":
            clip_f = float(arg)
        elif opt == "-s":
            s_max = float(arg)
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (input_pkl is None) or (prop_key is None):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool for CLAHE.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput GraphMCF: ' + input_pkl)
            print('\tProperty to equalize: ' + prop_key)
            print('\tCLAHE parameters:')
            print('\t\tNumber of neighbors and greyscales: ' + str(N))
            print('\t\tClipping factor: ' + str(clip_f))
            print('\t\tMaximum slope: ' + str(s_max))
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_clahe(input_pkl, prop_key, N, clip_f, s_max, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])