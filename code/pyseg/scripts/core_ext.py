"""

    Script for getting a core graph from a scalar filed and grow and extended one from the core

    Input:  - Original tomogram (density map)
            - Scalar field

    Output: - vtkPolyData with the core and extended graph
            - Segmentation of graphs

"""

__author__ = 'Antonio Martinez-Sanchez'


################# Package import

import sys
import time
import getopt
import disperse_io
import operator
from graph import GraphMCF
from globals import *
from factory import unpickle_obj
from factory import GraphsScalarMask

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
           '    -m <file_name>(optional): mask for disperse routines.\n' + \
           '    -n <file_name>(optional): input 3D image with the scalar field.\n' + \
           '    -d (optional): forces to re-update all DisPerSe data. \n' + \
           '    -r (optional): voxel resolution in nm of input data, default 1.68. \n' + \
           '    -g (optional): maximum geodesic distance for growing, default 0.' + \
           '    -C (optional): persistence threshold. \n' + \
           '    -S (optional): threshold for the scalar field. \n' + \
           '    -s (optional): number of sigmas used for segmentation.' + \
           '    -f <em, mrc, fits or vti>: segmentation output format (default mrc). \n' + \
           '    -v (optional): verbose mode activated.'

LOCAL_DEF_RES = 1.68
FIELD_NAME = 'field_mask'

################# Work routine

def do_core_grow(input_file, output_dir, fmt, scalar_file=None, mask_file=None, res=1, cut_t=None,
                 scalar_t=None, max_dist=0, f_update=False, sig=None, verbose=False):

    # Initialization
    if verbose:
        print('\tInitializing...')
    work_dir = output_dir + '/disperse'
    disperse = disperse_io.DisPerSe(input_file, work_dir)
    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
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
        disperse.mse(no_cut=False, inv=False)
    skel = disperse.get_skel()
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)

    # Build the PSD ArcGraph
    pkl_sgraph = work_dir + '/skel_graph.pkl'
    if f_update or (not os.path.exists(pkl_sgraph)):
        if verbose:
            print('\tBuilding graph...')
        graph = GraphMCF(skel, manifolds, density)
        graph.set_resolution(res)
        graph.build_from_skel()
        if verbose:
            print('\tAdding geometry...')
        graph.build_vertex_geometry()
        if verbose:
            print('\tPickling...')
        graph.pickle(pkl_sgraph)
        _, stem = os.path.split(input_file)
        stem, _ = os.path.splitext(stem)
        disperse_io.save_vtp(graph.get_vtp(), output_dir + '/' + stem + '_graph.vtp')
    else:
        if verbose:
            print('\tUnpickling graph...')
        graph = unpickle_obj(pkl_sgraph)

    if verbose:
        print('\tMasking with the scalar field...')
    field = disperse_io.load_tomo(scalar_file)
    factor = GraphsScalarMask(graph, field, FIELD_NAME)
    factor.gen_core_graph(scalar_t, operator.gt)
    factor.gen_ext_graph(max_dist)
    core_g = factor.get_core_graph()
    ext_g = factor.get_ext_graph()

    if verbose:
        print('\tSegmentation...')
    seg_core = core_g.print_vertices(property=DPSTR_CELL, th_den=sig)
    seg_ext = ext_g.print_vertices(property=DPSTR_CELL, th_den=sig)

    if verbose:
        print('\tStoring the result...')
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')
    disperse_io.save_vtp(core_g.get_vtp(av_mode=True), output_dir + '/' + stem + '_cor.vtp')
    disperse_io.save_vtp(ext_g.get_vtp(av_mode=True), output_dir + '/' + stem + '_ext.vtp')
    disperse_io.save_numpy(seg_core, output_dir + '/' + stem + '_cor_seg' + fmt)
    disperse_io.save_numpy(seg_ext, output_dir + '/' + stem + '_ext_seg' + fmt)


################# Main call

def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hvdi:o:m:n:r:S:C:s:g:f:")
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(2)

    input_file = ''
    output_dir = ''
    scalar_file = ''
    mask_file = ''
    f_update = False
    res = LOCAL_DEF_RES
    cut_t = None
    scalar_t = None
    fmt = '.mrc'
    verbose = False
    sig = None
    max_dist = 0
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
        elif opt == "-n":
            scalar_file = arg
        elif opt == "-d":
            f_update = True
        elif opt == "-r":
            res = float(arg)
        elif opt == "-C":
            cut_t = float(arg)
        elif opt == "-S":
            scalar_t = float(arg)
        elif opt == "-f":
            fmt = '.' + arg
        elif opt == "-s":
            sig = float(arg)
        elif opt == "-g":
            max_dist = float(arg)
        elif opt == "-v":
            verbose = True

    if (input_file == '') or (output_dir == ''):
        print(usage_msg)
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print('Running tool for getting core and extended graphs.')
            print('\tAuthor: ' + __author__)
            print('\tDate: ' + time.strftime("%c") + '\n')
            print('Options:')
            print('\tInput file: ' + input_file)
            print('\tOutput directory: ' + output_dir)
            if scalar_file == '':
                print('\tScalar field not used.')
            else:
                print('\tScalar field file: ' + scalar_file)
            if f_update:
                print('\tUpdate disperse: yes')
                if cut_t is not None:
                    print('\tPersistence threshold: ' + str(cut_t))
            else:
                print('\tUpdate disperse: no')
            print('\tResolution: ' + str(res) + ' nm')
            if scalar_t is not None:
                print('\tScalar field threshold threshold: ' + str(scalar_t))
            print('\tOutput segmentation format ' + fmt)
            print('\n')

        # Do the job
        if verbose:
            print('Starting...')
        do_core_grow(input_file, output_dir, fmt, scalar_file, mask_file, res, cut_t, scalar_t,
                     max_dist, f_update, sig, verbose)

        if verbose:
            print(cmd_name + ' successfully executed.')

if __name__ == "__main__":
    main(sys.argv[1:])