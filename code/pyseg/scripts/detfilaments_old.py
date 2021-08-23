"""

    Script for segmenting the filaments in a tomogram without assuming any geometric constrain
    (old version because it does not use MCF classes)

    Input:  - Original tomogram (density map)
            - Mask tomogram (0-fg, otherwise-bg)

    Output: - vtkPolyData with the filaments
            - Segmentation tomogram

"""

__author__ = 'Antonio Martinez-Sanchez'


################# Package import

import sys
import time
import getopt
import disperse_io
import operator
import factory
from graph import SkelGraph
from globals import *
from factory import unpickle_obj
from factory import ArcGraphFactory
from vtk_ext import vtkFilterRedundacyAlgorithm

try:
    import pickle as pickle
except:
    import pickle

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomogram> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input 3D density image in MRC, EM or FITS formats. \n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored. If it already exists then will be cleared before running the tool. \n' + \
           '    -m <file_name>(optional): input 3D image with the mask for data (0-foreground).\n' + \
           '    -d (optional): forces to re-update all DisPerSe data. \n' + \
           '    -r (optional): voxel resolution in nm of input data, default 1.68. \n' + \
           '    -C (optional): threshold for cutting low persistence structures. \n' + \
           '    -L (optional): threshold for low density maxima. \n' + \
           '    -H (optional): threshold for high density minima. \n' + \
           '    -R (optional): threshold for relevance. \n' + \
           '    -S (optional): threshold for length. \n' + \
           '    -f <em, mrc, fits or vti>: segmentation output format (default mrc). \n' + \
           '    -v (optional): verbose mode activated.'

LOCAL_DEF_RES = 1.68

################# Work routine

def do_det_filament(input_file, output_dir, fmt, mask_file=None, res=1, cut_t=None, low_t=None,
                    high_t=None, rel_t=None, len_t=None, f_update=False, verbose=False):

    # Initialization
    if verbose:
        print('\tInitializing...')
    work_dir = output_dir + '/disperse'
    disperse = disperse_io.DisPerSe(input_file, work_dir)
    # Down skeleton
    disperse.set_dump_arcs(-1)
    if cut_t is not None:
        disperse.set_cut(cut_t)
    if mask_file != '':
        disperse.set_mask(mask_file)
    density = disperse_io.load_tomo(input_file)

    # Disperse
    if verbose:
        print('\tRunning DisPerSe...')
    if f_update:
        disperse.clean_work_dir()
        disperse.mse(no_cut=False, inv=True)
    skel = disperse.get_skel()
    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)

    # Filtering the redundancy of the skeleton
    print('\tFiltering redundancy on DiSPerSe skeleton...')
    red_filt = vtkFilterRedundacyAlgorithm()
    red_filt.SetInputData(skel)
    red_filt.Execute()
    skel = red_filt.GetOutput()

    # Build the PSD ArcGraph
    pkl_sgraph = work_dir + '/skel_graph.pkl'
    if f_update or (not os.path.exists(pkl_sgraph)):
        if verbose:
            print('\tBuilding skeleton graph...')
        skel_graph = SkelGraph(skel)
        skel_graph.update()
        skel_graph.pickle(pkl_sgraph)
    else:
        if verbose:
            print('\tUnpickling PDS graphs...')
        skel_graph = unpickle_obj(pkl_sgraph)

    if verbose:
        print('\tThresholding the skeleton graph...')
    skel_graph.add_geometry(manifolds, density)
    if (low_t is not None) or (high_t is not None):
        skel_graph.find_critical_points()
    if low_t is not None:
        # skel_graph.threshold_maxima(STR_DENSITY_VERTEX, low_t, operator.gt)
        skel_graph.threshold_maxima(STR_DENSITY_VERTEX, low_t, operator.gt)
    if high_t is not None:
        skel_graph.threshold_minima(STR_DENSITY_VERTEX, low_t, operator.lt)

    if verbose:
        print('\tBuilding skeleton graph...')
    arc_graph = factory.build_arcgraph(skel_graph)

    disperse_io.save_vtp(arc_graph.get_vtp(), output_dir + '/hold.vtp')
    if verbose:
        print('\tThresholding the arcs graph...')
    if rel_t is not None:
        arc_graph.update_geometry(manifolds, density)
        arc_graph.compute_arc_relevance()
        arc_graph.threshold_arc(STR_ARC_RELEVANCE, rel_t, operator.gt)

    if verbose:
        print('\tBuilding the subgraphs network...')
    factor = ArcGraphFactory(arc_graph)
    network = factor.gen_netarcgraphs()

    if verbose:
        print('\tThresholding the network...')
    if len_t is not None:
        network.compute_diameters(resolution=res)
        network.threshold_subgraph(STR_GRAPH_DIAM, len_t, operator.gt)

    if verbose:
        print('\tSegmentation...')
    seg = arc_graph.print_arcs()

    if verbose:
        print('\tStoring the result...')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')
    disperse_io.save_vtp(skel_graph.get_vtp(), output_dir + '/' + stem + '_skel.vtp')
    disperse_io.save_vtp(arc_graph.get_vtp(), output_dir + '/' + stem + '_arc.vtp')
    disperse_io.save_numpy(seg, output_dir + '/' + stem + '_seg' + fmt)


################# Main call

def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hvdi:o:m:r:C:L:H:R:S:f")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = ''
    output_dir = ''
    mask_file = ''
    f_update = False
    res = LOCAL_DEF_RES
    cut_t = None
    low_t = None
    high_t = None
    rel_t = None
    len_t = None
    fmt = '.mrc'
    verbose = False
    for opt, arg in opts:
        if opt == '-h':
            print(usage_msg)
            print(help_msg)
            sys.exit()
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-m":
            mask_file = arg
        elif opt == "-d":
            f_update = True
        elif opt == "-r":
            res = float(arg)
        elif opt == "-C":
            cut_t = float(arg)
        elif opt == "-L":
            low_t = float(arg)
        elif opt == "-H":
            high_t = float(arg)
        elif opt == "-R":
            rel_t = float(arg)
        elif opt == "-S":
            len_t = float(arg)
        elif opt == "-f":
            fmt = arg
        elif opt == "-v":
            verbose = True

    if (input_file == '') or (output_dir == ''):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool for detecting filaments in a tomogram.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tOutput directory: ' + output_dir)
            if mask_file == '':
                print('\tMask not used.')
            else:
                print('\tMask file: ' + mask_file)
            if f_update:
                print('\tUpdate disperse: yes')
            else:
                print('\tUpdate disperse: no')
            print('\tResolution: ' + str(res) + ' nm')
            if cut_t is not None:
                print('\tPersistence threshold: ' + str(cut_t))
            if low_t is not None:
                print('\tLow density maxima threshold: ' + str(low_t))
            if high_t is not None:
                print('\tHigh density minima threshold: ' + str(high_t))
            if rel_t is not None:
                print('\tRelevance threshold: ' + str(rel_t))
            if len_t is not None:
                print('\tLength threshold: ' + str(len_t))
            print('\tOutput segmentation format ' + fmt)
            print('\n')

        # Do the job
        if verbose:
            print('Starting...')
        do_det_filament(input_file, output_dir, fmt, mask_file,  res, cut_t, low_t, high_t, rel_t,
                        len_t, f_update, verbose)

        if verbose:
            print(cmd_name + ' successfully executed.')

if __name__ == "__main__":
    main(sys.argv[1:])

