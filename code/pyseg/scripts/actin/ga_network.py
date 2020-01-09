"""

    Script for extracting the graphs for analyzing quantitatively a GA network

    Input:  - Density map tomogram
            - Segmentation tomogram of the synapse

    Output: - A graph

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import gc
import os
import time
import scipy
import operator
import numpy as np
import pyseg as ps
import scipy as sp
try:
    import cPickle as pickle
except:
    import pickle

########## Global variables

PI_2 = np.pi * 2.
MSK_OFF = 2 # voxels
STR_LOC_BET = 'loc_bet'
STR_LOC_BET_E = 'edge_loc_bet'

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/tomograms/marion/Clusters'

# Pre-processed density maps
in_tomo_l = (ROOT_PATH + '/in/fits/pre/20141124_W6t1_pre.fits',
             # ROOT_PATH + '/in/fits/pre/20150930_W3t1_pre.fits',
             )

# Masks
in_mask_l = (ROOT_PATH + '/in/fits/pre/20141124_W6t1_fmask_zcrop.fits',
             # ROOT_PATH + '/in/fits/pre/20150930_W3t1_fmask.fits',
             )
in_mask_st_l = (None, 
                # None,
               )

# Fields value tomograms
in_field_l = (ROOT_PATH + '/in/fits/pre/20141124_W6t1_cc.fits',
             # ROOT_PATH + '/in/fits/pre/20150930_W3t1_cc.fits',
             )

# Pickles (to skip GraphMCF computing)
update_hold = True
in_pkl_l = (ROOT_PATH + '/g_ga/20141124_W6t1_pre_hold.pkl',
            # ROOT_PATH + '/g_ga/20150930_W3t1_pre_hold.pkl',
            )

####### Output data

output_dir = ROOT_PATH+'/g_ga'

####### GraphMCF

s_sig = 1.0
csig_l = (0.1, 
          # 0.1,
          )
ang_rot = None # -96.9174
ang_tilt = None # 60
nstd_l = (3, 
          # 3,
          )
smooth = 3
res = 2.736 # nm/pix

######## Masking thresholds

mask_off = 1 # nm
cc_neigh = 5 # nm

####### Graph density thresholds

do_den = True
v_den = 0.0003 # nm^3
ve_ratio = 2
v_prop = ps.globals.STR_FIELD_VALUE # In None topological simplification
e_prop = ps.globals.STR_FIELD_VALUE # ps.globals.STR_VERT_DST
v_mode = 'low' # None
e_mode = 'low'
prop_topo = ps.globals.STR_FIELD_VALUE # None is ps.globals.STR_FIELD_VALUE

###### CLAHE

do_clahe = False
clahe_dst = 300 # nm

####### Field value thresholds

th_field_l = (0.35,
              # 0.01,) # 0.18,) # STR_FIELD_VALUE if no other field value is added (in_field_l)
              )
th_op = operator.lt

####### Betweeness decimation

bet_dec = 1

####### Filamentous-like filtering

do_fil = True
fl_mn_len = 50 # nm 50 (Marion)
fl_mx_len = 150 # nm
fl_avg_len = 5 # nm
fl_mx_ktt = 1.3 # 0.25*PI_2 # rad
fl_set = True

####### Global filtering

g_min_diam = 100 # nm
z_rang = [0, 90] # deg

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print 'Extracting GraphMCF from filamentous networks.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tDisPerSe persistence threshold (csig): ' + str(csig_l)
if ang_rot is not None:
    print 'Missing wedge edge compensation (rot, tilt): (' + str(ang_rot) + ', ' + str(ang_tilt) + ')'
print '\tSigma for gaussian pre-processing: ' + str(s_sig)
print '\tSigma for contrast enhancement: ' + str(nstd_l)
print '\tSkeleton smoothing factor: ' + str(smooth)
print '\tData resolution: ' + str(res) + ' nm/pixel'
print '\tOutput directory: ' + output_dir
if do_clahe:
    print '\tCLAHE maximum geodesic distance: ' + str(clahe_dst) + ' nm'
if do_den:
    print 'Graph density thresholds:'
    if v_prop is None:
        print '\tTarget vertex density (membranes) ' + str(v_den) + ' vertex/nm^3 for topological simplification'
    else:
        print '\tTarget vertex density (membranes) ' + str(v_den) + ' vertex/nm^3 for property ' + v_prop + ' with mode ' + v_mode
    print '\tTarget edge/vertex ratio (non membrane) ' + str(ve_ratio) + ' for property ' + e_prop + ' with mode ' + e_mode
    print '\tNeighbor size for cross-correlation: ' + str(cc_neigh) + ' nm'
if th_field_l is not None:
    print '\tField value global thresholding:'
    print '\t\t-Threshold: ' + str(th_field_l)
    print '\t\t-Operator: ' + str(th_op)
if bet_dec:
    print '\tBetweeness decimation by factor ' + str(bet_dec)
if do_fil:
    print '\tFilamentous like filter: '
    print '\t\t-Minimum length: ' + str(fl_mn_len) + ' nm'
    print '\t\t-Maximum length: ' + str(fl_mx_len) + ' nm'
    print '\t\t-Third curvature faction: ' + str(fl_mx_ktt) + ' rad'
    print '\t\t-Geometry subsampling distance: ' + str(fl_avg_len) + ' nm'
    if fl_set:
        print '\t\t-Store filaments set'
print ''

# Loop for processing the input data
print 'Running main loop: '
for (in_tomo, in_mask, in_mask_st, in_pkl, in_field, csig, nstd, th_field) in \
    zip(in_tomo_l, in_mask_l, in_mask_st_l, in_pkl_l, in_field_l, csig_l, nstd_l, th_field_l):

    print '\tComputing paths for ' + in_tomo + ' ...'
    f_path, f_fname = os.path.split(in_tomo)
    f_stem_pkl, f_ext = os.path.splitext(f_fname)
    input_file = output_dir + '/' + f_stem_pkl + '_g' + str(s_sig) + '.fits'
    _, f_stem = os.path.split(input_file)
    f_stem, _ = os.path.splitext(f_stem)

    if in_pkl is None:

        print '\tLoading mask: ' + in_mask
        mask = ps.disperse_io.load_tomo(in_mask).astype(np.bool)

        if in_mask_st is not None:
            print '\tAdding structures mask: ' + in_mask_st
            mask_st = ps.disperse_io.load_tomo(in_mask_st)
            mask_st = mask_st > mask_st.min()
            if MSK_OFF > 0:
                mask_st = scipy.ndimage.morphology.binary_dilation(mask_st, iterations=MSK_OFF)
            mask[mask_st] = True
        mask_i = np.invert(mask)

        print '\tLoading input data: ' + f_fname
        tomo = ps.disperse_io.load_tomo(in_tomo).astype(np.float32)

        print '\tSmoothing input tomogram (s=' + str(s_sig) + ')...'
        density = scipy.ndimage.filters.gaussian_filter(tomo, s_sig)
        density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1, mask=mask_i)

        print '\tStoring temporal tomograms for DisPerSe...'
        ps.disperse_io.save_numpy(tomo, output_dir + '/' + f_stem_pkl + '.vti')
        ps.disperse_io.save_numpy(density.transpose(), input_file)
        ps.disperse_io.save_numpy(density, output_dir + '/' + f_stem + '.vti')

        print '\tStoring temporal tomograms for VTK...'
        ps.disperse_io.save_numpy(density, output_dir + '/' + f_stem_pkl + '.vti')

        print '\tInitializing DisPerSe...'
        work_dir = output_dir + '/disperse'
        disperse = ps.disperse_io.DisPerSe(input_file, work_dir)
        try:
            disperse.clean_work_dir()
        except ps.pexceptions.PySegInputWarning as e:
            print e.get_message()

        in_hold_mask = work_dir + '/' + os.path.split(in_mask)[1]
        print '\tStroing hold DisPerSe mask in: ' + in_hold_mask
        ps.disperse_io.save_numpy(np.asarray(mask, dtype=np.float32).transpose(), in_hold_mask)

        # Manifolds for descending fields with the inverted image
        disperse.set_manifolds('J0a')
        # Down skeleton
        disperse.set_dump_arcs(-1)
        rcut = round(density[mask_i].std()*csig, 4)
        print '\tPersistence cut thereshold set to: ' + str(rcut) + ' grey level'
        disperse.set_cut(rcut)
        disperse.set_mask(in_hold_mask)
        disperse.set_smooth(smooth)

        print '\tRunning DisPerSe...'
        try:
            disperse.mse(no_cut=False, inv=False)
            skel = disperse.get_skel()
            manifolds = disperse.get_manifolds(no_cut=False, inv=False)
        except Exception:
            print 'WARNING: DisPerSe failed tomogram skipped!!!!'
            continue

        # Build the GraphMCF for the membrane
        print '\tBuilding MCF graph...'
        graph = ps.graph.GraphMCF(skel, manifolds, density)
        graph.build_from_skel(basic_props=False)
        graph.set_resolution(res)
        graph.filter_self_edges()
        graph.filter_repeated_edges()

        print '\t\tNumber of survivor vertices: ' + str(len(graph.get_vertices_list()))
        print '\tBuilding geometry...'
        graph.build_vertex_geometry()

        print '\tComputing vertices and edges properties...'
        graph.add_prop_inv(ps.globals.STR_FIELD_VALUE, edg=True)
        graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
        graph.compute_vertices_dst()
        graph.compute_edge_filamentness()
        graph.compute_edge_affinity()

        print '\tClean temporary data...'
        del density
        del mask
        del mask_i
        del tomo
        gc.collect()

        if do_clahe:
            print '\tCLAHE graph equalization...'
            graph.clahe_field_value_skel(clahe_dst, N=256, clip_f=100, s_max=4)
            graph.invert_prop(ps.globals.STR_FIELD_VALUE_EQ, ps.globals.STR_FIELD_VALUE_EQ+'_inv')
            # graph.add_prop_inv(ps.globals.STR_FIELD_VALUE, edg=True)

        print '\t\tStoring intermediate graph....'
        graph.pickle(output_dir + '/' + f_stem_pkl + '_hold.pkl')

        print '\tApplying general thresholds...'
        if ang_rot is not None:
            print '\tDeleting edges in MW area...'
            graph.filter_mw_edges(ang_rot, ang_tilt)

    else:
        print '\tLoading intermediate GraphMCF: ' + in_pkl
        graph = ps.factory.unpickle_obj(in_pkl)

    print '\tLoading mask: ' + in_mask
    mask = ps.disperse_io.load_tomo(in_mask).astype(np.bool)

    if in_mask_st is not None:
        print '\tAdding structures mask: ' + in_mask_st
        mask_st = ps.disperse_io.load_tomo(in_mask_st)
        mask_st = mask_st > mask_st.min()
        if MSK_OFF > 0:
            mask_st = scipy.ndimage.morphology.binary_dilation(mask_st, iterations=MSK_OFF)
        mask[mask_st] = True
    mask = np.invert(mask)

    if do_den:
        print '\tComputing graph global statistics (before simplification)...'
        nvv, nev, nepv = graph.compute_global_stat(mask=mask)
        print '\t\t-Vertex density: ' + str(nvv) + ' nm^3'
        print '\t\t-Edge density: ' + str(nev) + ' nm^3'
        print '\t\t-Edge/Vertex ratio: ' + str(nepv)

        print '\tGraph density simplification for vertices...'
        if (prop_topo is not None) and (prop_topo != ps.globals.STR_FIELD_VALUE):
            print '\t\tProperty used: ' + prop_topo
        graph.set_pair_prop(prop_topo)
        try:
            if prop_topo is not None:
                graph.graph_density_simp_ref(mask=np.asarray(mask, dtype=np.int), v_den=v_den,
		                                     v_prop=None, v_mode=v_mode)
            else:
                graph.graph_density_simp_ref(mask=np.asarray(mask, dtype=np.int), v_den=v_den,
		                                     v_prop=v_prop, v_mode=v_mode)
        except ps.pexceptions.PySegInputWarning as e:
            print 'WARNING: graph density simplification failed:'
            print '\t-' + e.get_message()

        print '\tGraph density simplification for edges...'
        nvv, nev, nepv = graph.compute_global_stat(mask=mask)
        if nepv > ve_ratio:
            e_den = nvv * ve_ratio
            hold_e_prop = e_prop
            graph.graph_density_simp_ref(mask=np.asarray(mask, dtype=np.int), e_den=e_den, e_prop=hold_e_prop, e_mode=e_mode, fit=True)
        else:
            print '\tWARNING: demanded ratio ' + str(ve_ratio) + ' could not be achieved (current is ' + str(nepv)

        print '\tComputing graph global statistics (after simplification)...'
        nvv, nev, nepv = graph.compute_global_stat(mask=mask)
        print '\t\t-Vertex density: ' + str(nvv) + ' nm^3'
        print '\t\t-Edge density:' + str(nev) + ' nm^3'
        print '\t\t-Edge/Vertex ratio: ' + str(nepv)

    if th_field is not None:
        print '\tField value global filtering:'
        if in_field is not None:
            print '\t\t-Adding field value: ' + str(in_field)
            field = ps.disperse_io.load_tomo(in_field)
            graph.add_scalar_field(field, 'cc', neigh=20, mode='max')
            print '\t\t-Filtering vertices...'
            graph.threshold_vertices('cc', th_field, th_op)
        else:
            graph.threshold_vertices(ps.disperse_io.STR_FIELD_VALUE, th_field, th_op)
            # graph.threshold_edges(ps.disperse_io.STR_FIELD_VALUE, th_field, operator.gt)

    print '\tCompute properties (2)...'
    # graph.compute_edge_curvatures()
    graph.compute_vertices_dst()
    # graph.compute_edge_filamentness()
    graph.compute_edges_length()
    graph.compute_edge_affinity()
    # graph.compute_edge_vectors()
    # graph.compute_vertex_vectors(30, key_dst=ps.disperse_io.STR_VERT_DST, fupdate=True)
    graph.compute_edge_zang()

    if update_hold:
        hold_pkl = output_dir + '/' + f_stem_pkl + '_hold.pkl'
        print '\tPickling hold GraphMCF: ' + hold_pkl
        graph.pickle(output_dir + '/' + f_stem_pkl + '_hold.pkl')

    if bet_dec > 1:
        print '\tBetweeness decimation by factor ' + str(bet_dec) + '...'
        graph_gt = ps.graph.GraphGT(graph)
        graph.bet_decimation(bet_dec, graph_gt, key_e=ps.globals.STR_FIELD_VALUE)

    if do_fil:
        print '\tFilamentous like filtering...'
        if fl_set:
            set_fils = graph.find_max_fil_persistence(fl_mn_len, fl_mx_len, fl_mx_ktt, fl_avg_len, gen_fils=True)
            out_set = output_dir+'/'+f_stem_pkl+'_set.vtp'
            print '\t\t-Storing filaments set in file: ' + out_set
            ps.disperse_io.save_vtp(set_fils.get_vtp(), out_set)
        else:
            graph.find_max_fil_persistence(fl_mn_len, fl_mx_len, fl_mx_apex, fl_avg_len, gen_fils=False)

    if z_rang is not None:
        print '\tFiltering edges with inclination in range:' + str(z_rang) +  ' deg'
        graph.threshold_edges(ps.disperse_io.STR_EDGE_ZANG, z_rang[0], operator.lt)
        graph.threshold_edges(ps.disperse_io.STR_EDGE_ZANG, z_rang[1], operator.gt)

    if g_min_diam is not None:
        print '\tThreshold by graph diameter (minimum): ' + str(g_min_diam) + ' nm'
        graph.compute_diameters(update=True)
        graph.compute_sgraph_relevance()
        graph.threshold_vertices(ps.globals.STR_GRAPH_DIAM, g_min_diam, operator.lt)

    print '\tSaving intermediate graphs...'
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + f_stem_pkl + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True), output_dir + '/' + f_stem_pkl + '_edges_2.vtp')
    # ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
    #                         output_dir + '/' + f_stem_pkl + '_sch.vtp')
    ps.disperse_io.save_numpy(graph.print_vertices(th_den=-1),
                              output_dir + '/' + f_stem_pkl + '_seg.vti')

    print '\tPickling the graph as: ' + f_stem_pkl + '.pkl'
    graph.pickle(output_dir + '/' + f_stem_pkl + '.pkl')

print 'Terminated. (' + time.strftime("%c") + ')'
