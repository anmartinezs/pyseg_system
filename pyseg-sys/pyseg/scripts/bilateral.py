"""

    Bilateral Filter for GraphMCF

    Input:  - GraphMCF (pickle file)

    Output: - Bilateral filtering stored in a vertex property

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import sys
import time
import getopt
from globals import *
from factory import unpickle_obj
from graph import GraphGT

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf> -o <dir_name>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <string>: input Graph MCF in a pickle file. \n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored.\n' + \
           '    -s <float>(optional): sigma for geodesic distance (default 3).' + \
           '    -r <float>(optional): sigma for radiometric distance (default 3).' + \
           '    -V <string>(optional): property used for vertex weighting.' + \
           '    -x(optional): if present vertex weights are inverted.' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_bilateral(input_file, output_dir, ss, sr, prop_v, vinv, verbose):

    if verbose:
        print '\tLoading the graph...'
    path, stem = os.path.split(input_file)
    stem, ext = os.path.splitext(stem)
    if ext == '.pkl':
        graph_mcf = unpickle_obj(input_file)
    else:
        print '\tERROR: ' + ext + ' is a non valid format.'
        sys.exit(4)

    if verbose:
        print '\tGetting the GT graph...'
    graph = GraphGT(graph_mcf)

    if verbose:
        print '\tApplying Bilateral Filter...'
    graph.bilateral(ss, sr, prop_v=prop_v, vinv=vinv)

    if verbose:
        print '\tAdding property to GraphMCF...'
    graph.add_prop_to_GraphMCF(graph_mcf, STR_BIL, up_index=True)

    if verbose:
        print '\tStoring the result in ' + output_dir
    _, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    graph_mcf.pickle(output_dir + '/' + stem + '.pkl')
    disperse_io.save_vtp(graph_mcf.get_vtp(av_mode=False, edges=True),
                         output_dir + '/' + stem + '_edges.vtp')
    disperse_io.save_vtp(graph_mcf.get_scheme_vtp(nodes=True, edges=True),
                         output_dir + '/' + stem + '_sch.vtp')

################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvxi:o:s:r:V:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = None
    output_dir = None
    prop_v = None
    vinv = False
    ss = 3.
    sr = 3.
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print usage_msg
            print help_msg
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-s":
            ss = float(arg)
        elif opt == "-r":
            sr = float(arg)
        elif opt == "-V":
            prop_v = arg
        elif opt == "-x":
            vinv = True
        elif opt == "-v":
            verbose = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(3)

    if (input_file is None) or (output_dir is None):
        print usage_msg
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print 'Running tool for computing PSM centrality.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + input_file
            print '\tOutput directory: ' + output_dir
            print '\tSigma geodesic: ' + str(ss)
            print '\tSigma radiometric: ' + str(sr)
            if prop_v is not None:
                if vinv:
                    print '\tVertex weighting property ' + prop_v + ' (inverted).'
                else:
                    print '\tVertex weighting property ' + prop_v + '.'
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_bilateral(input_file, output_dir, ss, sr, prop_v, vinv, verbose)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'


if __name__ == "__main__":
    main(sys.argv[1:])
