"""

    Script for extracting an analyzing a GraphMCF with a cleft
    (v3) 24.03.16 - Modification for not to compute filaments

    Input:  - Density map tomogram
            - Segmentation tomogram

    Output: - GraphMCF

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import time
import pyseg as ps
import scipy as sp
import os
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

########## Global variables

SEG_NAME ='cleft_seg'

# Labels
CLEFT_LBL = 1
PST_MB_LBL = 2
PRE_MB_LBL = 3
PST_CITO_LBL = 4
PRE_CITO_LBL = 5

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1/in/zd/bin2'
# ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1/in/test'

# Original density map
input_tomo_l = (ROOT_PATH+'/syn_14_14_bin2_rot_crop2.fits',
                )

# Segmentation tomograms
input_seg_l = (ROOT_PATH+'/syn_14_14_bin2_crop2_clft_seg.fits',
             )

####### Output data

output_dir = '/home/martinez/workspace/disperse/data/psd_an1/zd/pst_4/cleft_batch'
# output_dir = '/home/martinez/workspace/disperse/data/psd_an1/test/mb_graph'

####### GraphMCF

s_sig = 1.5
csig = 0.01
ang_rot = None # -96.9174
ang_tilt = None # 60
nstd = 3
smooth = 3
res = 0.684 # nm/pix
DILATE_NITER = 2 # pix

####### Graph density thresholds

v_den = 0.008 # nm^3
ve_ratio = 2
v_prop = None # ps.globals.STR_FIELD_VALUE # In None topological simplification
e_prop = ps.globals.STR_FIELD_VALUE_EQ # ps.globals.STR_VERT_DST
v_mode = None # 'low'
e_mode = 'low'
prop_topo = ps.globals.STR_FIELD_VALUE_EQ # None is ps.globals.STR_FIELD_VALUE

######## Masking thresholds
max_len = 15 # nm

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print 'Extracting GraphMCF for clefts in tomograms.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tDisPerSe persistence threshold (csig): ' + str(csig)
if ang_rot is not None:
    print 'Missing wedge edge compensation (rot, tilt): (' + str(ang_rot) + ', ' + str(ang_tilt) + ')'
print '\tSigma for gaussian pre-processing: ' + str(s_sig)
print '\tSigma for contrast enhancement: ' + str(nstd)
print '\tSkeleton smoothing factor: ' + str(smooth)
print '\tData resolution: ' + str(res) + ' nm/pixel'
print '\tOutput directory: ' + output_dir
print 'Graph density thresholds:'
if v_prop is None:
    print '\tTarget vertex density (membrane) ' + str(v_den) + ' vertex/nm^3 for topological simplification'
else:
    print '\tTarget vertex density (membrane) ' + str(v_den) + ' vertex/nm^3 for property ' + v_prop + ' with mode ' + v_mode
print '\tTarget edge/vertex ratio (non membrane) ' + str(ve_ratio) + ' for property ' + e_prop + ' with mode ' + e_mode
print ''

# Loop for processing the input data
print 'Running main loop: '
for (in_tomo, in_seg) in zip(input_tomo_l, input_seg_l):

    print '\tComputing paths for ' + in_tomo + ' ...'
    path, stem_tomo = os.path.split(in_tomo)
    stem_pkl, _ = os.path.splitext(stem_tomo)
    input_file = output_dir + '/' + stem_pkl + '_g' + str(s_sig) + '.fits'
    _, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)

    print '\tLoading input data: ' + stem_tomo
    tomo = ps.disperse_io.load_tomo(in_tomo)
    seg = ps.disperse_io.load_tomo(in_seg)

    print '\tComputing mask for DisPerSe...'
    maskh = (seg == PST_MB_LBL) + (seg == PRE_MB_LBL) + (seg == CLEFT_LBL)
    tomod = ps.disperse_io.seg_dist_trans(maskh.astype(np.bool)) * res
    maskh = np.ones(shape=seg.shape, dtype=np.int)
    maskh[DILATE_NITER:-DILATE_NITER, DILATE_NITER:-DILATE_NITER, DILATE_NITER:-DILATE_NITER] = 0
    mask = np.asarray(tomod > (max_len + 2*DILATE_NITER*res), dtype=np.int)
    maskh += mask
    mask = np.asarray(maskh > 0, dtype=np.float)
    input_msk = output_dir + '/' + stem + '_mask.fits'
    ps.disperse_io.save_numpy(mask.transpose(), input_msk)
    mask_mbs = ((seg == PST_MB_LBL) + (seg == PRE_MB_LBL)).astype(np.bool)

    print '\tSmoothing input tomogram (s=' + str(s_sig) + ')...'
    density = sp.ndimage.filters.gaussian_filter(tomo, s_sig)
    density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1)
    ps.disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')
    ps.disperse_io.save_numpy(density.transpose(), input_file)

    print '\tInitializing DisPerSeg...'
    work_dir = output_dir + '/disperse'
    disperse = ps.disperse_io.DisPerSe(input_file, work_dir)
    disperse.clean_work_dir()
    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    # Down skeleton
    disperse.set_dump_arcs(-1)
    # disperse.set_nsig_cut(nsig)
    rcut = round(density[mask.astype(np.bool)].std()*csig, 4)
    print '\tPersistence cut thereshold set to: ' + str(rcut) + ' grey level'
    disperse.set_cut(rcut)
    disperse.set_mask(input_msk)
    disperse.set_smooth(smooth)

    print '\tRunning DisPerSe...'
    disperse.mse(no_cut=False, inv=False)
    skel = disperse.get_skel()
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)

    # Build the GraphMCF for the membrane
    print '\tBuilding MCF graph...'
    graph = ps.mb.MbGraphMCF(skel, manifolds, density, seg)
    graph.set_resolution(res)
    graph.build_from_skel(basic_props=False)
    graph.filter_self_edges()
    graph.filter_repeated_edges()

    print '\tFiltering nodes close to mask border...'
    mask = sp.ndimage.morphology.binary_dilation(mask, iterations=DILATE_NITER)
    for v in graph.get_vertices_list():
        x, y, z = graph.get_vertex_coords(v)
        if mask[int(round(x)), int(round(y)), int(round(z))]:
            graph.remove_vertex(v)
    print '\tBuilding geometry...'
    graph.build_vertex_geometry()

    print '\tCLAHE on filed_value_inv property...'
    graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.clahe_field_value(max_geo_dist=50, N=256, clip_f=100., s_max=4.)

    print '\tComputing vertices and edges properties...'
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()
    graph.add_prop_inv(prop_topo, edg=True)
    graph.compute_edge_affinity()

    print '\tApplying general thresholds...'
    if ang_rot is not None:
        print '\tDeleting edges in MW area...'
        graph.filter_mw_edges(ang_rot, ang_tilt)

    print '\tComputing graph global statistics in membranes (before simplification)...'
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_mbs)
    print '\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3'
    print '\t\t-Edge density: ' + str(round(nev,5)) + ' nm^3'
    print '\t\t-Edge/Vertex ratio: ' + str(round(nepv,5))

    print '\tGraph density simplification for vertices with membranes as reference...'
    if prop_topo != ps.globals.STR_FIELD_VALUE:
        print '\t\tProperty used: ' + prop_topo
        graph.set_pair_prop(prop_topo)
    try:
        graph.graph_density_simp_ref(mask=np.asarray(mask_mbs, dtype=np.int), v_den=v_den,
                                     v_prop=v_prop, v_mode=v_mode)
    except ps.pexceptions.PySegInputWarning as e:
        print 'WARNING: graph density simplification failed:'
        print '\t-' + e.get_message()

    print '\tGraph density simplification for edges in post membrane...'
    mask_pst = (seg == PST_MB_LBL) * (~mask)
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_pst)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_pst, dtype=np.int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)

    print '\tGraph density simplification for edges in pre membrane...'
    mask_pre = (seg == PRE_MB_LBL) * (~mask)
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_pre)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_pre, dtype=np.int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)

    print '\tGraph density simplification for edges in the PSD...'
    mask_psd = (seg == PST_CITO_LBL) * (~mask)
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_psd)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_psd, dtype=np.int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print '\tWARNING: demanded ratio ' + str(nepv) + \
              ' could not be achieved (current is ' + str(nepv)

    print '\tGraph density simplification for edges in the AZ...'
    mask_az = (seg == PRE_CITO_LBL) * (~mask)
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_az)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_az, dtype=np.int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print '\tWARNING: demanded ratio ' + str(nepv) + \
              ' could not be achieved (current is ' + str(nepv)

    print '\tGraph density simplification for edges in the Cleft...'
    mask_clft = (seg == CLEFT_LBL) * (~mask)
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_clft)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_clft, dtype=np.int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print '\tWARNING: demanded ratio ' + str(nepv) + \
              ' could not be achieved (current is ' + str(nepv)

    print '\tComputing graph global statistics (after simplification)...'
    nvv, nev_mb, nepv_mb = graph.compute_global_stat(mask=mask)
    _, nev_pst, nepv_pst = graph.compute_global_stat(mask=mask_pst)
    _, nev_pre, nepv_pre = graph.compute_global_stat(mask=mask_pre)
    _, nev_psd, nepv_psd = graph.compute_global_stat(mask=mask_psd)
    _, nev_az, nepv_az = graph.compute_global_stat(mask=mask_az)
    _, nev_cl, nepv_cl = graph.compute_global_stat(mask=mask_clft)
    print '\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3'
    print '\t\t-Edge density (pst-membrane):' + str(round(nev_pst,5)) + ' nm^3'
    print '\t\t-Edge density (pre-membrane):' + str(round(nev_pre,5)) + ' nm^3'
    print '\t\t-Edge density (PSD):' + str(round(nev_psd,5)) + ' nm^3'
    print '\t\t-Edge density (AZ):' + str(round(nev_az,5)) + ' nm^3'
    print '\t\t-Edge density (Cleft):' + str(round(nev_cl,5)) + ' nm^3'
    print '\t\t-Edge/Vertex ratio (pst-membrane): ' + str(round(nepv_pst,5))
    print '\t\t-Edge/Vertex ratio (pst-membrane): ' + str(round(nepv_pre,5))
    print '\t\t-Edge/Vertex ratio (PSD): ' + str(round(nepv_psd,5))
    print '\t\t-Edge/Vertex ratio (AZ): ' + str(round(nepv_az,5))
    print '\t\t-Edge/Vertex ratio (Cleft): ' + str(round(nepv_cl,5))

    print '\tComputing graph properties (2)...'
    # graph.compute_edge_curvatures()
    graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()
    graph.compute_edge_affinity()

    print '\tAdding segmentation...'
    graph.add_scalar_field_nn(seg, SEG_NAME)

    print '\tSaving intermediate graphs...'
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + stem + '_edges_2.vtp')
    ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                            output_dir + '/' + stem + '_sch.vtp')

    print '\tPickling the graph as: ' + stem_pkl + '.pkl'
    # ps.disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')
    graph.pickle(output_dir + '/' + stem_pkl + '.pkl')

print 'Terminated. (' + time.strftime("%c") + ')'
