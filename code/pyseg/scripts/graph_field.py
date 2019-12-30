"""

    Script for inserting GraphMCF properties from a scalar field

    Input:  - GraphMCF instance (pkl)
            - Scalar field (tomogram)

    Output: - GraphMCF with the props added

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
from globals import *
from factory import unpickle_obj
from pyseg import disperse_io

try:
    import cPickle as pickle
except:
    import pickle

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomo> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: GraphMCF pickle, this file will be overwritten. \n' + \
           '    -s <file_name>: tomogram with the scalar field.' + \
           '    -p <string>: key name for the added property.' + \
           '    -m <string>(optional): interpolation mode, valid: manifolds, nhood, local' \
           '                           and none (default)' + \
           '    -n <int>(optional): neighbourhood size (in nm) for ''nhood'' mode ' + \
           '                        (default 5nm).' + \
           '    -o <string>(optional): operation applied in manifolds or nhood modes, valid' + \
           '                           sum (default), mean, median, min and max.' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_graph_field(input_file, scalar_file, prop_key, mode, neigh, oper, verbose=False):

    # Initialization
    if verbose:
        print '\tLoading graph...'
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph = unpickle_obj(input_file)

    if verbose:
        print '\tInserting scalar field property as %s...' + prop_key
    if mode == 'manifolds':
        graph.add_scalar_field(disperse_io.load_tomo(scalar_file), prop_key,
                               manifolds=True, mode=oper)
    else:
        if mode == 'nhood':
            graph.add_scalar_field(disperse_io.load_tomo(scalar_file), prop_key,
                                   manifolds=False, neigh=neigh, mode=oper)
        elif mode == 'none':
            graph.add_scalar_field_nn(disperse_io.load_tomo(scalar_file), prop_key)
        else:
            graph.add_scalar_field(disperse_io.load_tomo(scalar_file), prop_key,
                                   manifolds=False)

    if verbose:
        print '\tStoring the result...'
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph.pickle(input_file)
    disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                         path + '/' + stem + '_edges.vtp')
    disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                         path + '/' + stem + '_sch.vtp')
    if verbose:
        print '\tFile ' + input_file + ' overwritten.'


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvi:s:p:m:n:o:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = None
    scalar_file = None
    prop_key = None
    mode = 'none'
    neigh = 5.0
    oper = 'sum'
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print usage_msg
            print help_msg
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-s":
            scalar_file = arg
        elif opt == "-p":
            prop_key = arg
        elif opt == "-m":
            mode = arg
        elif opt == "-n":
            neigh = float(arg)
        elif opt == "-o":
            oper = arg
        elif opt == "-v":
            verbose = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(3)

    if (input_file is None) or (scalar_file is None) or (prop_key is None):
        print usage_msg
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print 'Running tool for getting the graph mcf of a tomogram.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + input_file
            print '\tScalar field: ' + scalar_file
            print '\tProperty name: ' + prop_key
            print '\tInterpolation mode:' + mode
            if mode == 'nhood':
                print '\t\tCubic neighbourhood size: ' + str(neigh) + 'nm'
            if mode != 'local':
                print '\t\tOperator: ' + oper
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_graph_field(input_file, scalar_file, prop_key, mode, neigh, oper, verbose=False)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'


if __name__ == "__main__":
    main(sys.argv[1:])