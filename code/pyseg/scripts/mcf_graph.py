"""

    Script for getting the graph MCF from a tomogram

    Input:  - Original tomogram (density map)
            - Mask (optional)

    Output: - GraphMCF object in a pkl file
            - vtkPolyData with the membrane and the attached structures

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
from pyseg.graph import GraphMCF
from pyseg.globals import *
import scipy as sp
from pyseg import disperse_io

try:
    import cPickle as pickle
except:
    import pickle

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomo> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input 3D density image in MRC, EM or FITS formats. \n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored.\n' + \
           '    -m <file_name>(optional): mask for disperse routines. If none it is computed' + \
           ' automatically.\n' + \
           '    -r <float>(optional): voxel resolution in nm of input data, default 1.68. \n' + \
           '    -a <float>(optional): tilt axis rotation angle (in deg). \n' + \
           '    -b <float>(optional): maximum tilt angle (default 90 deg).\n' + \
           '    -C <float>(optional): persistence threshold for DisPerSe. \n' + \
           '    -N <int>(optional): number of sigmas for DisPerSe''s persistence threshold' + \
           '                   (this is an alternative to option ''C'').' + \
           '    -S <int>(optional): DisPerSe skeleton smoothing.' + \
           '    -s <float>(optional): sigma value for Gaussian smoothing filter for input tomogram.' + \
           '    -c (optional): enable robustness computation. \n' + \
           '    -w <float>(optional): weighting for z dimension so as to compensate missing ' + \
           '                          wedge for distances measurement (default 1).' + \
           '    -f <em, mrc, fits or vti>: segmentation output format (default mrc). \n' + \
           '    -v (optional): verbose mode activated.'

LOCAL_DEF_RES = 1.68
DILATE_NITER = 1

################# Work routine

def do_mcf_graph(input_file, output_dir, mask_file=None, res=1, tilt_rot=None, tilt_ang=90,
              cut_t=None, rob=False, nsig_t=None, gsig=None, w_z=1, smooth=0, verbose=False):

    # Pre-processing
    if gsig is not None:
        if verbose:
            print '\tGaussian filtering for input data...'
        input_img = disperse_io.load_tomo(input_file)
        flt_img = sp.ndimage.filters.gaussian_filter(input_img, gsig)
        flt_img = lin_map(flt_img, lb=0, ub=1)
        path, stem = os.path.split(input_file)
        stem_pkl, _ = os.path.splitext(stem)
        disperse_io.save_numpy(input_img, output_dir + '/' + stem_pkl + '.vti')
        input_file = output_dir + '/' + stem_pkl + '_g' + str(gsig) + '.fits'
        disperse_io.save_numpy(flt_img.transpose(), input_file)

    # Initialization
    if verbose:
        print '\tInitializing...'
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    if stem_pkl is None:
        stem_pkl = stem
    work_dir = output_dir + '/disperse'
    disperse = disperse_io.DisPerSe(input_file, work_dir)
    disperse.clean_work_dir()
    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    # Down skeleton
    disperse.set_dump_arcs(-1)
    if cut_t is not None:
        disperse.set_cut(cut_t)
    elif nsig_t is not None:
        disperse.set_nsig_cut(nsig_t)
    if rob:
        disperse.set_robustness(True)
    disperse.set_mask(mask_file)
    disperse.set_smooth(smooth)
    density = disperse_io.load_tomo(input_file)

    # Disperse
    if verbose:
        print '\tRunning DisPerSe...'
    disperse.mse(no_cut=False, inv=False)
    skel = disperse.get_skel()
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)

    # Build the GraphMCF for the membrane
    if verbose:
        print '\tBuilding MCF graph...'
    graph = GraphMCF(skel, manifolds, density)
    graph.set_resolution(res)
    graph.build_from_skel(basic_props=False)
    graph.filter_self_edges()
    graph.filter_repeated_edges()
    if tilt_rot is not None:
        print '\tDeleting edges in MW area...'
        graph.filter_mw_edges(tilt_rot, tilt_ang)
    # Filter nodes close to mask border
    mask = disperse_io.load_tomo(mask_file)
    mask = sp.ndimage.morphology.binary_dilation(mask, iterations=DILATE_NITER)
    for v in graph.get_vertices_list():
        x, y, z = graph.get_vertex_coords(v)
        if mask[int(round(x)), int(round(y)), int(round(z))]:
            graph.remove_vertex(v)
    graph.build_vertex_geometry()

    if verbose:
        print '\tComputing graph properties...'
    graph.compute_edges_length(SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.compute_edges_length(SGT_EDGE_LENGTH_W, 1, 1, w_z, False)
    graph.compute_edges_length(SGT_EDGE_LENGTH_WTOTAL, 1, 1, w_z, True)
    # graph.compute_edges_sim()
    # graph.compute_edges_integration(field=False)
    graph.compute_edge_filamentness()
    graph.compute_vertices_dst()
    graph.add_prop_inv(STR_FIELD_VALUE)
    graph.compute_edge_affinity()
    graph.compute_edge_curvatures()
    # graph.compute_full_per(0.1)
    # graph.compute_sgraph_relevance()

    if verbose:
        print '\tStoring the result as ' + output_dir + '/' + stem_pkl + '.pkl'
    disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')
    disperse_io.save_numpy(lin_map(density, lb=0, ub=1),
                           output_dir + '/' + stem + '_inv.vti')
    graph.pickle(output_dir + '/' + stem_pkl + '.pkl')
    disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                         output_dir + '/' + stem + '_edges.vtp')
    disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                         output_dir + '/' + stem + '_edges_2.vtp')
    disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                         output_dir + '/' + stem + '_sch.vtp')

################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvci:o:m:r:a:b:s:w:C:N:S:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = ''
    output_dir = ''
    mask_file = ''
    res = LOCAL_DEF_RES
    cut_t = None
    nsig_t = None
    gsig = None
    tilt_rot = None
    tilt_ang = 90
    rob = False
    w_z = 1
    smooth = 0
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
        elif opt == "-r":
            res = float(arg)
        elif opt == "-a":
            tilt_rot = float(arg)
        elif opt == "-b":
            tilt_ang = float(arg)
        elif opt == "-C":
            cut_t = float(arg)
        elif opt == "-N":
            nsig_t = float(arg)
        elif opt == "-s":
            gsig = float(arg)
        elif opt == "-c":
            rob = True
        elif opt == "-w":
            w_z = float(arg)
        elif opt == '-S':
            smooth = int(arg)
        elif opt == "-v":
            verbose = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(3)

    if (input_file == '') or (output_dir == ''):
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
            if mask_file != '':
                print '\tMask file: ' + mask_file
            print '\tOutput directory: ' + output_dir
            if gsig is not None:
                print '\tPre-processing gaussian with sigma ' + str(gsig)
            if cut_t is not None:
                print '\tPersistence threshold: ' + str(cut_t)
            elif nsig_t is not None:
                print '\tPersistence number of sigmas threshold: ' + str(nsig_t)
            print '\tN times for skeleton smoothing: ' + str(smooth)
            print '\tResolution: ' + str(res) + ' nm/vox'
            if tilt_rot is not None:
                print '\tMissing wedge rotation angle ' + str(tilt_rot) + ' deg'
                print '\tMaximum tilt angle ' + str(tilt_ang) + ' deg'
            if rob:
                print '\tRobustness will be computed.'
            print '\tZ dimension weighting: ' + str(w_z)
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_mcf_graph(input_file, output_dir, mask_file, res, tilt_rot, tilt_ang, cut_t, rob,
                     nsig_t, gsig, w_z, smooth, verbose)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'


if __name__ == "__main__":
    main(sys.argv[1:])