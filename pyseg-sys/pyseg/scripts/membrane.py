"""

    Script for getting the graph of segmented membrane an the structures attached to it

    Input:  - Original tomogram (density map)
            - Membrane segmentation tomogram

    Output: - vtkPolyData with the membrane and the attached structures
            - Segmentation of the graphs

"""

__author__ = 'Antonio Martinez-Sanchez'


# ################ Package import

import sys
import time
import getopt
import operator
from graph import GraphMCF
from graph import GraphGT
from globals import *
from factory import unpickle_obj
from factory import GraphsSurfMask
from factory import short_path_accum

try:
    import cPickle as pickle
except:
    import pickle

################# Main declarations

_, cmd_name = os.path.split(__file__)

usage_msg = 'Basic Usage: ' + cmd_name + ' -i <input_tomo> -n <memb_seg_tomo> -o <output_dir> \n' + \
            'Run \'' + cmd_name + ' -h\' for help and see additional parameters.'
help_msg = '    -i <file_name>: input 3D density image in MRC, EM or FITS formats. \n' + \
           '    -n <file_name>: input 3D image with the membrane segmentation (1-fg, 0-bg).\n' + \
           '    -o <dir_name>: Name of the directory where output and intermediate data will be ' + \
           'stored. If it already exists then will be cleared before running the tool. \n' + \
           '    -m <file_name>(optional): mask for disperse routines. If none it is computed' + \
           ' automatically.\n' + \
           '    -d (optional): forces to re-update all DisPerSe data. \n' + \
           '    -r (optional): voxel resolution in nm of input data, default 1.68. \n' + \
           '    -g (optional): maximum geodesic distance for growing, default 0.' + \
           '    -t (optional): membrane thickness in nm, default 0.' + \
           '    -p (optional): membrane polarity, it controls which side is inspected.' + \
           'Valid entries: + (default) or -. ' + \
           '    -a (optional): tilt axis rotation angle (in deg). \n' + \
           '    -b (optional): maximum tilt angle (default 90 deg).\n' + \
           '    -C (optional): persistence threshold. \n' + \
           '    -L (optional): threshold for low density maxima. \n' + \
           '    -H (optional): threshold for high density minima. \n' + \
           '    -R (optional): threshold of robustness. \n' + \
           '    -s (optional): number of sigmas used for segmentation.' + \
           '    -f <em, mrc, fits or vti>: segmentation output format (default mrc). \n' + \
           '    -v (optional): verbose mode activated.'

LOCAL_DEF_RES = 1.68
SEG_NAME = 'seg_mask'
MASK_TH = 0.5
TOMO_BORDER_NVOX = 7
MASK_OFFSET = 5

################# Work routine

def do_mb_seg(input_file, seg_file, output_dir, fmt, mask_file=None, res=1, mb_thick=0,
              pol=STR_SIGN_P, tilt_rot=None, tilt_ang=90, cut_t=None, low_t=None, high_t=None,
              rob_t=None, max_dist=0, f_update=False, sig=None, verbose=False):
    # Initialization
    if verbose:
        print '\tInitializing...'
    path, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)
    work_dir = output_dir + '/disperse'
    disperse = disperse_io.DisPerSe(input_file, work_dir)
    if f_update:
        disperse.clean_work_dir()
    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    # Down skeleton
    disperse.set_dump_arcs(-1)
    if cut_t is not None:
        disperse.set_cut(cut_t)
    if rob_t is not None:
        disperse.set_robustness(True)
    # Getting distance and density fields
    mb_seg = disperse_io.load_tomo(seg_file)
    x_min = TOMO_BORDER_NVOX
    x_max = mb_seg.shape[0] - TOMO_BORDER_NVOX
    y_min = TOMO_BORDER_NVOX
    y_max = mb_seg.shape[1] - TOMO_BORDER_NVOX
    z_min = TOMO_BORDER_NVOX
    z_max = mb_seg.shape[2] - TOMO_BORDER_NVOX
    mb_seg = crop_cube(mb_seg, (x_min, x_max), (y_min, y_max), (z_min, z_max))
    surf, cloud = disperse_io.gen_surface_cloud(mb_seg, purge_ratio=1, cloud=True)
    sign_field = disperse_io.signed_dist_cloud(cloud, mb_seg.shape)
    sign_field = crop_cube(sign_field, (x_min, x_max), (y_min, y_max), (z_min, z_max),
                           np.max(sign_field))
    dist_field = disperse_io.seg_dist_trans(mb_seg)
    if mask_file != '':
        disperse.set_mask(mask_file)
    else:
        # Automatic generation of the mask
        x_m_min = MASK_OFFSET
        x_m_max = mb_seg.shape[0] - MASK_OFFSET
        y_m_min = MASK_OFFSET
        y_m_max = mb_seg.shape[1] - MASK_OFFSET
        z_m_min = MASK_OFFSET
        z_m_max = mb_seg.shape[2] - MASK_OFFSET
        mask = dist_field > ((max_dist / res) + MASK_OFFSET)
        mask_file = disperse.get_working_dir() + '/' + stem + '_mask.fits'
        mask = crop_cube(mask, (x_m_min, x_m_max), (y_m_min, y_m_max), (z_m_min, z_m_max), 1)
        disperse_io.save_numpy(mask.astype(np.int16).transpose(), mask_file)
        disperse.set_mask(mask_file)
    dist_field = crop_cube(dist_field, (x_min, x_max), (y_min, y_max), (z_min, z_max),
                           np.max(dist_field))
    density = disperse_io.load_tomo(input_file)

    # Disperse
    if verbose:
        print '\tRunning DisPerSe...'
    if f_update:
        disperse.mse(no_cut=False, inv=False)
    skel = disperse.get_skel()
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)

    # Build the GraphMCF for the membrane
    pkl_sgraph = work_dir + '/skel_graph.pkl'
    if f_update or (not os.path.exists(pkl_sgraph)):
        if verbose:
            print '\tBuilding graph...'
        graph = GraphMCF(skel, manifolds, density)
        graph.set_resolution(res)
        graph.build_from_skel(basic_props=False)
        if verbose:
            print '\tPickling...'
        graph.pickle(pkl_sgraph)
        _, stem = os.path.split(input_file)
        stem, _ = os.path.splitext(stem)
        disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                             output_dir + '/' + stem + '_graph.vtp')
    else:
        if verbose:
            print '\tUnpickling graph...'
        graph = unpickle_obj(pkl_sgraph)

    if verbose:
        print '\tPostprocessing the graph...'
    if low_t is not None:
        if verbose:
            print '\t\tLow vertex...'
        graph.threshold_vertices(STR_FIELD_VALUE, low_t, operator.gt)
    if high_t is not None:
        if verbose:
            print '\t\tHigh edge...'
        graph.threshold_edges(STR_FIELD_VALUE, high_t, operator.gt)
    if rob_t is not None:
        if verbose:
            print '\t\tRobustness...'
        graph.threshold_vertices(DPSTR_ROBUSTNESS, rob_t, operator.lt)
    # TODO: Add graph relevance filter
    # Post processing
    graph.filter_self_edges()
    graph.filter_repeated_edges()
    # if tilt_rot is not None:
        # graph.filter_mw_edges(tilt_rot, tilt_ang)

    if verbose:
        print '\tGetting membrane and attached structures graphs...'
    factor = GraphsSurfMask(graph, surf, sign_field, dist_field, pol, mb_thick)
    disperse_io.save_numpy(sign_field, output_dir + '/hold.vti')
    disperse_io.save_numpy(dist_field, output_dir + '/hold2.vti')
    factor.gen_core_graph()
    factor.gen_ext_graph(max_dist, keep_anchors=True)
    memb_g = factor.get_core_graph()
    # memb_g.filter_self_edges()
    # memb_g.filter_repeated_edges()
    att_g = factor.get_ext_graph()
    # att_g.filter_self_edges()
    # att_g.filter_repeated_edges()
    if tilt_rot is not None:
        memb_g.filter_mw_edges(tilt_rot, tilt_ang)
        att_g.filter_mw_edges(tilt_rot, tilt_ang)

    if verbose:
        print '\tComputing graph properties'
    memb_g.compute_diameters()
    att_g.compute_diameters()
    att_g.build_vertex_geometry()
    att_g.compute_sgraph_relevance()

    print 'Test accumulation'
    anchors = factor.get_anchor_vertices()
    end_nodes = factor.get_end_vertices()
    key_id = att_g.add_prop('mb_type', 'int', 1, 0)
    anchors_ref = list()
    for a in anchors:
        if att_g.get_vertex(a.get_id()) is not None:
            att_g.set_prop_entry_fast(key_id, (1,), a.get_id(), 1)
            anchors_ref.append(a)
    end_nodes_ref = list()
    for e in end_nodes:
        if att_g.get_vertex(e.get_id()) is not None:
            att_g.set_prop_entry_fast(key_id, (2,), e.get_id(), 1)
            end_nodes_ref.append(e)
    short_path_accum(att_g, anchors_ref, end_nodes_ref, 'paths_acc', STR_FIELD_VALUE)

    att_g.threshold_vertices('paths_acc', 0.5, operator.lt)
    skel_nodes = att_g.get_skel_vtp(mode='edge')
    g_filter = gauss_FE(skel_nodes, None, sigma=1.25, size=density.shape)
    disperse_io.save_numpy(g_filter, output_dir + '/hold.vti')

    if verbose:
        print '\tSegmentation...'
    memb_g.build_vertex_geometry()
    seg_core = memb_g.print_vertices(property=STR_GRAPH_ID, th_den=sig)
    mb_thick_vx = mb_thick / res
    dist_field = sign_field < np.ceil(mb_thick / res)
    dist_field *= sign_field > np.floor((-1) * (mb_thick / res))
    seg_core *= dist_field.astype(seg_core.dtype)
    seg_ext = att_g.print_vertices(property=STR_RAND_ID, th_den=sig)
    if pol == STR_SIGN_P:
        dist_field = sign_field > np.ceil(mb_thick_vx)
        seg_ext *= dist_field.astype(seg_ext.dtype)
    else:
        dist_field = sign_field < np.floor((-1) * mb_thick_vx)
        seg_ext *= dist_field.astype(seg_ext.dtype)

    if verbose:
        print '\tGenerating GT graphs'
    memb_gt = GraphGT(memb_g)
    memb_gt.betweenness(mode='both')
    att_gt = GraphGT(att_g)
    att_gt.betweenness(mode='both')

    if verbose:
        print '\tStoring the result...'
    disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')
    disperse_io.save_vtp(surf, output_dir + '/' + stem + '_surf.vtp')
    memb_g.pickle(output_dir + '/' + stem + '_mb.pkl')
    att_g.pickle(output_dir + '/' + stem + '_att.pkl')
    disperse_io.save_vtp(memb_g.get_vtp(av_mode=True, edges=True),
                         output_dir + '/' + stem + '_mb_hold.vtp')
    disperse_io.save_vtp(att_g.get_vtp(av_mode=True, edges=True),
                         output_dir + '/' + stem + '_att_hold.vtp')
    disperse_io.save_vtp(memb_g.get_scheme_vtp(nodes=True, edges=True),
                         output_dir + '/' + stem + '_mb_sch.vtp')
    disperse_io.save_vtp(att_g.get_scheme_vtp(nodes=True, edges=True),
                         output_dir + '/' + stem + '_att_sch.vtp')
    disperse_io.save_numpy(seg_core, output_dir + '/' + stem + '_mb_seg' + fmt)
    disperse_io.save_numpy(seg_ext, output_dir + '/' + stem + '_att_seg' + fmt)
    memb_gt.draw(v_color=SGT_BETWEENNESS, rm_vc=(0, 1), cmap_vc='jet',
                 e_color=SGT_BETWEENNESS, rm_ec=(0, 1), cmap_ec='hot',
                 v_size=STR_FIELD_VALUE, rm_vs=(5, 0),
                 e_size=STR_FIELD_VALUE, rm_es=(3, 0),
                 output=output_dir + '/' + stem + '_mb.pdf')
    memb_gt.save(output_dir + '/' + stem + '_mb.gt')
    att_gt.draw(v_color=SGT_BETWEENNESS, rm_vc=(0, 1), cmap_vc='hot',
                e_color=None, rm_ec=(0, 1), cmap_ec='hot',
                v_size=None, rm_vs=(8, 0),
                e_size=None, rm_es=(3, 0),
                output=output_dir + '/' + stem + '_att.pdf')
    att_gt.save(output_dir + '/' + stem + '_att.gt')


################# Main call

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvdi:o:m:n:r:t:p:a:b:C:L:H:R:s:g:f:")
    except getopt.GetoptError:
        print usage_msg
        sys.exit(2)

    input_file = ''
    output_dir = ''
    seg_file = ''
    mask_file = ''
    f_update = False
    res = LOCAL_DEF_RES
    cut_t = None
    low_t = None
    high_t = None
    rob_t = None
    tilt_rot = None
    tilt_ang = 90
    fmt = '.mrc'
    verbose = False
    sig = None
    max_dist = 0
    pol = STR_SIGN_P
    thick = 0
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
        elif opt == "-n":
            seg_file = arg
        elif opt == "-d":
            f_update = True
        elif opt == "-r":
            res = float(arg)
        elif opt == "-t":
            thick = float(arg)
        elif opt == "-p":
            pol = arg
        elif opt == "-a":
            tilt_rot = float(arg)
        elif opt == "-b":
            tilt_ang = float(arg)
        elif opt == "-C":
            cut_t = float(arg)
        elif opt == "-L":
            low_t = float(arg)
        elif opt == "-H":
            high_t = float(arg)
        elif opt == "-R":
            rob_t = float(arg)
        elif opt == "-f":
            fmt = '.' + arg
        elif opt == "-s":
            sig = float(arg)
        elif opt == "-g":
            max_dist = float(arg)
        elif opt == "-v":
            verbose = True
        else:
            print 'Unknown option ' + opt
            print usage_msg
            sys.exit(3)

    if (input_file == '') or (output_dir == '') or (seg_file == ''):
        print usage_msg
        sys.exit(2)
    else:
        # Print init message
        if verbose:
            print 'Running tool for getting graphs from membrane segmentation.'
            print '\tAuthor: ' + __author__
            print '\tDate: ' + time.strftime("%c") + '\n'
            print 'Options:'
            print '\tInput file: ' + input_file
            print '\tOutput directory: ' + output_dir
            if f_update:
                print '\tUpdate disperse: yes'
                if cut_t is not None:
                    print '\tPersistence threshold: ' + str(cut_t)
                if rob_t is not None:
                    print '\tRobustness threshold: ' + str(rob_t)
            else:
                print '\tUpdate disperse: no'
            print '\tResolution: ' + str(res) + ' nm/vox'
            print '\tMembrane thickness ' + str(thick) + ' nm'
            print '\tMaximum distance ' + str(max_dist)
            print '\tMembrane polarity ' + pol
            if tilt_rot is not None:
                print '\tMissing wedge rotation angle ' + str(tilt_rot) + ' deg'
                print '\tMaximum tilt angle ' + str(tilt_ang) + ' deg'
            if low_t is not None:
                print '\tLow density maxima threshold: ' + str(low_t)
            if high_t is not None:
                print '\tHigh density minima threshold: ' + str(high_t)
            if sig is not None:
                print '\tNo sigmas for segmentation: ' + str(sig)
            print '\tOutput segmentation format ' + fmt
            print ''

        # Do the job
        if verbose:
            print 'Starting...'
        do_mb_seg(input_file, seg_file, output_dir, fmt, mask_file, res, thick, pol, tilt_rot,
                  tilt_ang, cut_t, low_t, high_t, rob_t, max_dist, f_update, sig, verbose)

        if verbose:
            print cmd_name + ' successfully executed. (' + time.strftime("%c") + ')'


if __name__ == "__main__":
    main(sys.argv[1:])