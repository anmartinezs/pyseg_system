"""

    From a mask with sources and sinks computes the max flow network within a GraphMCF

    Input:  - GraphMCF (pickle file)
            - Mask (1-sources, 2-sinks, othwise-background)

    Output: - Max flow network stored as edge GraphMCF property

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

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <string>: input Graph MCF in a pickle file. \n' + \
           '    -m <string>: mask file tomogram: 1-sources, 2-sinks.' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored.\n' + \
           '    -w <string_key>: property used for edge weighting.' + \
           '    -x (optional): for inverting weighting values.' + \
           '    -a <string>(optional): algorithm, valid: ''ek'', ''pr'' (default) or ' \
           '    ''bk'' (see graph_tool doc).' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_max_flow(input_file, output_dir, mask_file, prop_w, einv, alg, verbose):

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
        print '\tLoading the mask file...'
    mask = disperse_io.load_tomo(mask_file)
    graph_mcf.threshold_edges_in_mask(mask)
    graph_mcf.simp_vertices(0)

    if verbose:
        print '\tGetting the GT graph...'
    graph_gt = GraphGT(graph_mcf)

    if verbose:
        print '\tSolving max flow problem...'
    graph_gt.compute_max_flow(graph_mcf, mask, prop_w, einv, alg)

    if verbose:
        print '\tAssign the clusters to the graph...'
    if alg == 'ek':
        graph_gt.add_prop_to_GraphMCF(graph_mcf, STR_MFLOW_EK, up_index=True)
    elif alg == 'bk':
        graph_gt.add_prop_to_GraphMCF(graph_mcf, STR_MFLOW_BK, up_index=True)
    else:
        graph_gt.add_prop_to_GraphMCF(graph_mcf, STR_MFLOW_PR, up_index=True)
    graph_gt.add_prop_to_GraphMCF(graph_mcf, STR_FLOW_SS, up_index=True)

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
        opts, args = getopt.getopt(argv, "hvxi:o:w:m:a:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = None
    output_dir = None
    mask_file = None
    prop_w = None
    einv = False
    alg = 'pr'
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
        elif opt == "-m":
            mask_file = arg
        elif opt == "-a":
            if (arg == 'ek') or (arg == 'pr') or (arg == 'bk'):
                alg = arg
            else:
                print 'Unknown argument for -a ' + arg
                print usage_msg
                sys.exit(3)
        elif opt == "-w":
            prop_w = arg
        elif opt == "-x":
            einv = True
        elif opt == "-v":
            verbose = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(3)

    if (input_file is None) or (output_dir is None) or (mask_file is None)\
            or (prop_w is None):
        print usage_msg
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print 'Running tool computing max flow network.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + input_file
            print '\tMask file: ' + mask_file
            print '\tOutput directory: ' + output_dir
            if einv:
                print '\tEdge weighting property (inverted): ' + prop_w
            else:
                print '\tEdge weighting property: ' + prop_w
            if alg == 'ek':
                print '\tAlgorithm: Edmonds-Karp'
            elif alg == 'bk':
                print '\tAlgorithm: Boykov-Kolmogorov'
            else:
                print '\tAlgorithm: Push relabel'
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_max_flow(input_file, output_dir, mask_file, prop_w, einv, alg, verbose)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'


if __name__ == "__main__":
    main(sys.argv[1:])
