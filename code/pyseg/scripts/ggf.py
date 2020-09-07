"""

    Computes Geodesic Gaussian Filter (GGF) for GraphMCF

    Input:  - GraphMCF (pickle file)

    Output: - GGF stored in a vertex property

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import sys
import time
import getopt
from globals import *
from factory import unpickle_obj
from graph import GraphGT
from pyseg import disperse_io

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf> -o <dir_name>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <string>: input Graph MCF in a pickle file. \n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored.\n' + \
           '    -s <float>(optional): sigma for gauss (default 3).' + \
           '    -V <string>(optional): property used for vertex weighting.' + \
           '    -E <string>(optional): property used for edge weighting.' + \
           '    -x(optional): if present vertex weights are inverted.' + \
           '    -y(optional): if present edge weights are inverted.' + \
           '    -w(optional): if present vertex weights are normalized to [0, 1].' + \
           '    -z(optional): if present edge weights are normalized to [0, 1].' + \
           '    -e(optional): if present energy normalization is activated.' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_ggf(input_file, output_dir, prop_v, prop_e, vinv, einv, vnorm, enorm, sig, energy,
           verbose):

    if verbose:
        print('\tLoading the graph...')
    path, stem = os.path.split(input_file)
    stem, ext = os.path.splitext(stem)
    if ext == '.pkl':
        graph_mcf = unpickle_obj(input_file)
    else:
        print('\tERROR: ' + ext + ' is a non valid format.')
        sys.exit(4)

    if verbose:
        print('\tGetting the GT graph...')
    graph = GraphGT(graph_mcf)

    if verbose:
        print('\tApplying GGF...')
    graph.ggf(sig, prop_v, prop_e, vinv, einv, vnorm, enorm, energy)

    if verbose:
        print('\tAdding property to GraphMCF...')
    graph.add_prop_to_GraphMCF(graph_mcf, STR_GGF, up_index=True)

    if verbose:
        print('\tStoring the result in ' + output_dir)
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
        opts, args = getopt.getopt(argv, "hvxywezi:o:s:V:E:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = None
    output_dir = None
    prop_v = None
    prop_e = None
    vinv = False
    einv = False
    vnorm = False
    enorm = False
    sig = 3.
    verbose = False
    energy = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-s":
            sig = float(arg)
        elif opt == "-V":
            prop_v = arg
        elif opt == "-E":
            prop_e = arg
        elif opt == "-x":
            vinv = True
        elif opt == "-y":
            einv = True
        elif opt == "-w":
            vnorm = True
        elif opt == "-z":
            enorm = True
        elif opt == "-e":
            energy = True
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if (input_file is None) or (output_dir is None):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool for computing PSM centrality.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tOutput directory: ' + output_dir)
            print('\tSigma: ' + str(sig))
            if prop_v is not None:
                if vinv:
                    if vnorm:
                        print('\tVertex weighting property ' + prop_v + ' (inverted and normalized).')
                    else:
                        print('\tVertex weighting property ' + prop_v + ' (inverted).')
                else:
                    if vnorm:
                        print('\tVertex weighting property ' + prop_v + '(normalized).')
                    else:
                        print('\tVertex weighting property ' + prop_v + '.')
            if prop_e is not None:
                if einv:
                    if enorm:
                        print('\tEdge weighting property ' + prop_e + ' (inverted and normalized).')
                    else:
                        print('\tEdge weighting property ' + prop_e + ' (inverted).')
                else:
                    if enorm:
                        print('\tEdge weighting property ' + prop_e + '(normalized).')
                    else:
                        print('\tEdge weighting property ' + prop_e + '.')
            if energy:
                print('\tEnergy normalization activated.')
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_ggf(input_file, output_dir, prop_v, prop_e, vinv, einv, vnorm, enorm, sig, energy,
               verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])
