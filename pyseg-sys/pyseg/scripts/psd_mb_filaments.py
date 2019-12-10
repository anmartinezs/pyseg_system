"""

    Script for segmenting and quantifying the filaments attached to a post-synaptic membrane

    Input:  - Original tomogram (density map)
            - Synaptic membrane segmentation tomogram (pre and post membranes)

    Output: - TODO: STILL PENDING TO BE DEFINED
            - vtkPolyData
            - Statistical data
            - Segmentation data

"""

__author__ = 'Antonio Martinez-Sanchez'

################# Package import

import vtk
import sys
import os
import time
import getopt
import disperse_io
import pexceptions
import numpy as np
# import matplotlib.pyplot as plt
from psd import *
from factory import FilFactory
from factory import unpickle_obj
from vtk_ext import vtkFilterRedundacyAlgorithm

try:
    import cPickle as pickle
except:
    import pickle


################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomogram> -s <input_seg> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help'
help_msg = '    -i <file_name>: input 3D density image in MRC, EM or FITS formats. \n' + \
           '    -s <file_name>: input 3D image with pre (1) and post-membranes (2) segmented . \n' + \
           '    -o <dir_name>: Name of the directory where data will be stored. If it already \n' +\
           '                   exists it will be cleared previously. \n' + \
           '    -m <file_name>(optional): input 3D image with the mask for data (zeros is foreground).\n' + \
           '    -d (optional): forces to re-update all DisPerSe data. \n' + \
           '    -c (optional): thresholding for cutting low persistence structures. \n' + \
           '    -r (optional): voxel resolution in nm of input data, default 1.88. \n' + \
           '    -t (optional): membrane thickness in nm, default 5. \n' + \
           '    -p (optional): post-synaptic thickness in nm, default 60. \n' + \
           '    -q (optional): pre-synaptic thickness in nm, default 10. \n' + \
           '    -e (optional): density threshold (n. sigs.) for storing geometry mask.' + \
           '    -b (optional): number of bins for the filaments length frequency histogram.' + \
           '    -v (optional): verbose mode activated, disabled by default. \n'

################# Work routine

def do_psd_mb_filament(input_file, seg_file, output_dir, mask_file, p_cut, resolution, mb_thickness,
                       post_thickness, pre_thickness, f_update=False, th_density=None, n_bins=0,
                       verbose=False):

    # Initialization
    if verbose:
        print '\tInitializing...'
    work_dir = output_dir + '/disperse'
    disperse = disperse_io.DisPerSe(input_file, work_dir)
    # Down skeleton
    disperse.set_dump_arcs(-1)
    if p_cut is not None:
        disperse.set_cut(p_cut)
    if mask_file != '':
        disperse.set_mask(mask_file)
    density = disperse_io.load_tomo(input_file)
    seg = disperse_io.load_tomo(seg_file)
    # disperse_io.save_vti(disperse_io.numpy_to_vti(density), 'hold.vti', output_dir)

    # Disperse
    if verbose:
        print '\tRunning DisPerSe...'
    if f_update:
        disperse.clean_work_dir()
        disperse.mse(no_cut=False, inv=True)
    skel = disperse.get_skel()
    # disperse.set_vertex_as_min(mode=True)
    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)
    if density.shape != seg.shape:
        if seg.shape != manifolds.shape:
            seg = seg.transpose()
        else:
            density = density.transpose()

    # Filtering the redundancy of the skeleton
    print '\tFiltering redundancy on DiSPerSe skeleton...'
    red_filt = vtkFilterRedundacyAlgorithm()
    red_filt.SetInputData(skel)
    red_filt.Execute()
    skel = red_filt.GetOutput()

    # Build the PSD ArcGraph
    pkl_psd = work_dir + '/psd.pkl'
    if f_update or (not os.path.exists(pkl_psd)):
        if verbose:
            print '\tGetting PSD graphs...'
        psd = PsdSynap(skel, manifolds, density, seg)
        psd.set_resolution(resolution)
        psd.set_memb_thickness(mb_thickness)
        psd.set_max_dist_post(post_thickness)
        psd.set_max_dist_pre(pre_thickness)
        psd.build_agraphs(geometry=False)
        psd.pickle(pkl_psd)
    else:
        if verbose:
            print '\tUnpickling PDS graphs...'
        psd = unpickle_obj(pkl_psd)

    if verbose:
        print '\tExtracting Post-synaptic graph...'
    arc_graph = psd.get_post_arc_graph()
    arc_graph.add_geometry(manifolds, density)
    arc_graph.find_critical_points()
    arc_graph.copy_varcs_prop(psd.get_post_skel_graph())

    # Extracting the filaments
    if verbose:
        print '\tExtracting the filaments...'
    f_factory = FilFactory(arc_graph)
    network = f_factory.inter_point_filaments(psd.get_post_anchors())

    # Analyzing the filaments
    if verbose:
        print '\tAnalyzing the filaments...'
    # network.arc_redundancy()
    network.vertex_redundancy()

    # Storing the result
    if verbose:
        print '\tStoring the result...'
    writer = vtk.vtkXMLPolyDataWriter()
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    out_fname = output_dir + '/' + stem
    writer.SetFileName(out_fname + '_fnet.vtp')
    vtk_ver = vtk.vtkVersion().GetVTKVersion()
    poly_f = network.get_vtp('filament')
    if int(vtk_ver[0]) < 6:
        writer.SetInput(poly_f)
    else:
        writer.SetInputData(poly_f)
    if writer.Write() == 1:
        if verbose:
            print '\t\tFile %s stored.' % writer.GetFileName()
    else:
        error_msg = 'Error writing %s.' % writer.GetFileName()
        raise pexceptions.PySegInputError(expr='do_psd_mb_filament (psd_mb_filaments)', msg=error_msg)
    del writer
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_fname + '_anet.vtp')
    poly_a = network.get_vtp('arc')
    if int(vtk_ver[0]) < 6:
        writer.SetInput(poly_a)
    else:
        writer.SetInputData(poly_a)
    if writer.Write() == 1:
        if verbose:
            print '\t\tFile %s stored.' % writer.GetFileName()
    else:
        error_msg = 'Error writing %s.' % writer.GetFileName()
        raise pexceptions.PySegInputError(expr='do_psd_mb_filament (psd_mb_filaments)', msg=error_msg)
    del writer
    writer = vtk.vtkXMLImageDataWriter()
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    out_fname = output_dir + '/' + stem
    writer.SetFileName(out_fname + '_mask.vti')
    # network.add_geometry(manifolds, density)
    img = network.print_mask(img=None, th_den=th_density, manifold=manifolds, density=density)
    if int(vtk_ver[0]) < 6:
        writer.SetInput(disperse_io.numpy_to_vti(img))
    else:
        writer.SetInputData(disperse_io.numpy_to_vti(img))
    if writer.Write() == 1:
        if verbose:
            print '\t\tFile %s stored.' % writer.GetFileName()
    else:
        error_msg = 'Error writing %s.' % writer.GetFileName()
        raise pexceptions.PySegInputError(expr='do_psd_mb_filament (psd_mb_filaments)', msg=error_msg)

    # Do analysis
    if n_bins > 0:
        print '\tGenerating analysis'
        lengths = network.get_fil_lengths()
        hist = np.histogram(lengths, n_bins)
        len_fname = output_dir + '/' + stem + '_len.pkl'
        hist_fname = output_dir + '/' + stem + '_hist.pkl'
        pkl_f = open(len_fname, 'w')
        try:
            pickle.dump(lengths, pkl_f)
        finally:
            pkl_f.close()
        pkl_f = open(hist_fname, 'w')
        try:
            pickle.dump(hist, pkl_f)
        finally:
            pkl_f.close()
        # plt.plot(hist[0])
        # plt.title("Filaments length histogram")
        # plt.xlabel("Length")
        # plt.ylabel("Frequency")
        # plt.show()


################# Main call

def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hvdi:s:o:m:c:r:t:p:q:e:b:v")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = ''
    output_dir = ''
    seg_file = ''
    mask_file = ''
    f_update = False
    res = 1.88
    mb_t = 5.0
    cut = None
    post_t = 60
    pre_t = 10
    verbose = False
    th_density = None
    n_bins = 0
    for opt, arg in opts:
        if opt == '-h':
            print usage_msg
            print help_msg
            sys.exit()
        elif opt == "-d":
            f_update = True
        elif opt == "-i":
            input_file = arg
        elif opt == "-o":
            output_dir = arg
        elif opt == "-r":
            res = float(arg)
        elif opt == "-t":
            mb_t = float(arg)
        elif opt == "-s":
            seg_file = arg
        elif opt == "-m":
            mask_file = arg
        elif opt == "-c":
            cut = float(arg)
        elif opt == "-p":
            post_t = float(arg)
        elif opt == "-q":
            pre_t = float(arg)
        elif opt == "-e":
            th_density = float(arg)
        elif opt == '-b':
            n_bins = int(arg)
        elif opt == "-v":
            verbose = True

    if (input_file == '') or (seg_file == '') or (output_dir == ''):
        print usage_msg
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print 'Running for analyzing the filaments attached to a post-synaptic membrane.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + input_file
            print '\tSegmentation file: ' + seg_file
            print '\tOutput directory: ' + output_dir
            if mask_file == '':
                print '\tMask not used.'
            else:
                print '\tMask file: ' + mask_file
            if f_update:
                print '\tUpdate disperse: yes'
            else:
                print '\tUpdate disperse: no'
            print '\tResolution: ' + str(res) + ' nm'
            print '\tMembrane thickness: ' + str(mb_t) + ' nm'
            print '\tPostsynaptic thickness: ' + str(post_t) + ' nm'
            print '\tPresynaptic thickness: ' + str(pre_t) + ' nm'
            print '\tPersistence threshold: ' + str(cut)
            print '\n'

        # Do the job
        if verbose:
            print 'Starting...'
        do_psd_mb_filament(input_file, seg_file, output_dir, mask_file, cut, res, mb_t, post_t,
                           pre_t, f_update, th_density, n_bins, verbose)

        if verbose:
            print cmd_name + ' successfully executed.'

if __name__ == "__main__":
    main(sys.argv[1:])
