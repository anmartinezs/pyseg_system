"""

    Script for drawing GraphMCF instances as graphs

    Input:  - File with a GraphMCF (.pkl)

    Output: - Output view (interactive window or .pdf file)

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
from graph import GraphGT
from globals import *
from factory import unpickle_obj
from factory.utils import graph_draw
import operator

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_graph_mcf>\n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input Graph MCF in a pickle file. \n' + \
           '    -o <output_file>(optional): Name of the output file in PDF. If not activated \n' + \
           'is going to be printed out in a interactive window. \n' + \
           '    -w <key>(optional): key string for the property used for node color. \n' + \
           '    -x <key>(optional): key string for the property used for edge color. \n' + \
           '    -y <key>(optional): key string for the property used for node diameter. \n' + \
           '    -z <key>(optional): key string for the property used for edge thickness. \n' + \
           '    -m <key>(optional): key string color map used (see scipy doc, default hot).' + \
           '    -l <key>(optional): key string for the layout scheme (see graph_tool doc,' + \
           '    default sfdp)' + \
           '    -v (optional): verbose mode activated.'

################# Work routine

def do_gt_draw(input_file, output_file=None, v_color=None, v_size=None, e_color=None, e_size=None,
               cmap='hot', layout='sfdp', verbose=False):

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
        print('\tGenerating the GT graph...')
    graph_gt = GraphGT(graph_mcf).get_gt()

    # Get largest component
    l = gt.label_largest_component(graph_gt)
    graph_gt = gt.GraphView(graph_gt, vfilt=l)
    # gt.make_maximal_planar(graph_gt)

    if verbose:
        print('\tDrawing the graph...')
    # Parsing layouts
    if layout == 'sfdp':
        pos = gt.sfdp_layout(graph_gt)
    elif layout == 'fruchterman':
        pos = gt.fruchterman_reingold_layout(graph_gt)
    elif layout == 'arf':
        pos = gt.arf_layout(graph_gt)
    elif layout == 'random':
        pos = gt.random_layout(graph_gt)
    elif layout == 'radial':
        print('\tRadial tree layout selected -> computing tree root...')
        bet, _ = gt.betweenness(graph_gt)
        bet_max_id = np.argmax(bet.get_array())
        graph_gt.vertex_properties[SGT_BETWEENNESS] = bet
        print('\tVertex betweenness updated.')
        pos = gt.radial_tree_layout(graph_gt, root=bet_max_id)
    graph_draw(graph_gt, pos,
               v_color, (0, 1), None,
               e_color, (0, 1), None,
               v_size, (0, 5),
               e_size, (0, 3),
               output_file)


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvri:o:w:x:y:z:m:l:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = ''
    output_file = None
    v_color = None
    v_size = None
    e_color = None
    e_size = None
    cmap = 'hot'
    layout = 'sfdp'
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_file = arg
        elif opt == "-w":
            v_color = arg
        elif opt == "-x":
            e_color = arg
        elif opt == "-y":
            v_size = arg
        elif opt == "-z":
            e_size = arg
        elif opt == "-m":
            cmap = arg
        elif opt == "-l":
            layout = arg
        elif opt == "-v":
            verbose = True
        else:
            print('Unknown option ' + opt)
            print(usage_msg)
            sys.exit(3)

    if input_file == '':
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool drawing a graph.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            if output_file is not None:
                print('\tOutput file: ' + output_file)
            else:
                print('\tInteractive mode activated.')
            if v_color is not None:
                print('\tProperty for vertices color: ' + v_color)
            if v_size is not None:
                print('\tProperty for vertices diameters: ' + v_size)
            if e_color is not None:
                print('\tProperty for vertices color: ' + e_color)
            if e_size is not None:
                print('\tProperty for vertices diameters: ' + e_size)
            if (v_color is not None) or (e_color is not None):
                print('\tColor map: ' + cmap)
            print('\tLayout scheme: ' + layout)
            print('')

        # Do the job
        if verbose:
            print('Starting...')
        do_gt_draw(input_file, output_file, v_color, v_size, e_color, e_size, cmap, layout, verbose)

        if verbose:
            print(cmd_name + ' successfully executed. (' + time.strftime("%c") + ')')


if __name__ == "__main__":
    main(sys.argv[1:])