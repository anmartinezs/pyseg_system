"""

    Script for extracting an analyzing a GraphMCF with a segmented membrane
    (v3) 18.11.15 - Modification for not to compute filaments

    Input:  - Density map tomogram
            - Segmentation tomogram

    Output: - Connectors clusters

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import time
import pyseg as ps
import scipy as sp
import os
import numpy as np
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

# Membrane segmentation: 1-mb, 2-cito, 3-ext
SEG_MB = 1
SEG_MB_IN = 2
SEG_MB_OUT = 3
SEG_TAG = '_seg'

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/pool/pool-lucic2/antonio/workspace/vesicles'

# Input STAR file with segmentations
in_star = ROOT_PATH + '/pre/bin2/con_11_2_bin2_hold.star'

####### Output data

output_dir = ROOT_PATH+'/graph/hold'

####### GraphMCF

s_sig = 1.5
# nsig = 0.01
csig = 0.01
ang_rot = None # -96.9174
ang_tilt = None # 60
nstd = 3
smooth = 3
res = 0.684 # nm/pix
mb_dst_off = 0. # nm
DILATE_NITER = 2 # pix

######## Masking thresholds
max_len = 30 # nm

####### Graph density thresholds

v_den = 0.006 # nm^3
ve_ratio = 2
v_prop = None # ps.globals.STR_FIELD_VALUE # In None topological simplification
e_prop = ps.globals.STR_FIELD_VALUE_EQ # ps.globals.STR_VERT_DST
v_mode = None # 'low'
e_mode = 'low'
prop_topo = ps.globals.STR_FIELD_VALUE_EQ # None is ps.globals.STR_FIELD_VALUE

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print('Extracting GraphMCF and NetFilament objects from tomograms')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
# print '\tDisPerSe persistence threshold (nsig): ' + str(nsig)
print('\tSTAR file with the segmentations: ' + str(in_star))
print('\tDisPerSe persistence threshold (csig): ' + str(csig))
if ang_rot is not None:
    print('Missing wedge edge compensation (rot, tilt): (' + str(ang_rot) + ', ' + str(ang_tilt) + ')')
print('\tSigma for gaussian pre-processing: ' + str(s_sig))
print('\tSigma for contrast enhancement: ' + str(nstd))
print('\tSkeleton smoothing factor: ' + str(smooth))
print('\tData resolution: ' + str(res) + ' nm/pixel')
print('\tMask offset: ' + str(mb_dst_off) + ' nm')
print('\tOutput directory: ' + output_dir)
print('Graph density thresholds:')
if v_prop is None:
    print('\tTarget vertex density (membrane) ' + str(v_den) + ' vertex/nm^3 for topological simplification')
else:
    print('\tTarget vertex density (membrane) ' + str(v_den) + ' vertex/nm^3 for property ' + v_prop + ' with mode ' + v_mode)
print('\tTarget edge/vertex ratio (non membrane) ' + str(ve_ratio) + ' for property ' + e_prop + ' with mode ' + e_mode)
print('')

print('Paring input star file...')
star = ps.sub.Star()
star.load(in_star)
in_seg_l = star.get_column_data('_psSegImage')
star.add_column('_psGhMCFPickle')

# Loop for processing the input data
print('Running main loop: ')
for (row, input_seg) in zip(list(range(star.get_nrows())), in_seg_l):

    seg_tag_pos = input_seg.find(SEG_TAG)
    if seg_tag_pos < 0:
        print('WARNING: no cropped sub-volume found for segmentation: ' + input_seg)
    input_tomo = input_seg[:seg_tag_pos] + input_seg[seg_tag_pos+len(SEG_TAG):]
    print('\tSub-volume to process found: ' + input_tomo)
    print('\tComputing paths for ' + input_tomo + ' ...')
    path, stem_tomo = os.path.split(input_tomo)
    stem_pkl, _ = os.path.splitext(stem_tomo)
    input_file = output_dir + '/' + stem_pkl + '_g' + str(s_sig) + '.fits'
    _, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)

    print('\tLoading input data: ' + stem_tomo)
    tomo = ps.disperse_io.load_tomo(input_tomo)
    segh = ps.disperse_io.load_tomo(input_seg)

    print('\tComputing distance, mask and segmentation tomograms...')
    tomod = ps.disperse_io.seg_dist_trans(segh == SEG_MB) * res
    maskh = np.ones(shape=segh.shape, dtype=int)
    maskh[DILATE_NITER:-DILATE_NITER, DILATE_NITER:-DILATE_NITER, DILATE_NITER:-DILATE_NITER] = 0
    mask = np.asarray(tomod > (max_len + mb_dst_off + 2*DILATE_NITER*res), dtype=int)
    maskh += mask
    mask = np.asarray(maskh > 0, dtype=float)
    input_msk = output_dir + '/' + stem + '_mask.fits'
    ps.disperse_io.save_numpy(mask.transpose(), input_msk)
    if mb_dst_off > 0:
        seg = np.zeros(shape=segh.shape, dtype=segh.dtype)
        seg[tomod < mb_dst_off] = SEG_MB
        seg[(seg == 0) & (segh == SEG_MB_IN)] = SEG_MB_IN
        seg[(seg == 0) & (segh == SEG_MB_OUT)] = SEG_MB_OUT
        tomod = ps.disperse_io.seg_dist_trans(seg == SEG_MB) * res
    else:
        seg = segh
    mask_den = np.asarray(tomod <= mb_dst_off, dtype=bool)

    print('\tSmoothing input tomogram (s=' + str(s_sig) + ')...')
    density = sp.ndimage.filters.gaussian_filter(tomo, s_sig)
    density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1)
    ps.disperse_io.save_numpy(tomo, output_dir + '/' + stem_pkl + '.vti')
    ps.disperse_io.save_numpy(density.transpose(), input_file)
    ps.disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')

    print('\tInitializing DisPerSe...')
    work_dir = output_dir + '/disperse'
    disperse = ps.disperse_io.DisPerSe(input_file, work_dir)
    try:
        disperse.clean_work_dir()
    # except ps.pexceptions.PySegInputWarning as e:
    #     print e.get_message()
    except Warning:
        print('Jol!!!')

    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    # Down skeleton
    disperse.set_dump_arcs(-1)
    # disperse.set_nsig_cut(nsig)
    rcut = round(density[mask_den].std()*csig, 4)
    print('\tPersistence cut thereshold set to: ' + str(rcut) + ' grey level')
    disperse.set_cut(rcut)
    disperse.set_mask(input_msk)
    disperse.set_smooth(smooth)

    print('\tRunning DisPerSe...')
    disperse.mse(no_cut=False, inv=False)
    skel = disperse.get_skel()
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)

    # Build the GraphMCF for the membrane
    print('\tBuilding MCF graph...')
    graph = ps.mb.MbGraphMCF(skel, manifolds, density, seg)
    graph.set_resolution(res)
    graph.build_from_skel(basic_props=False)
    graph.filter_self_edges()
    graph.filter_repeated_edges()

    print('\tFiltering nodes close to mask border...')
    mask = sp.ndimage.morphology.binary_dilation(mask, iterations=DILATE_NITER)
    for v in graph.get_vertices_list():
        x, y, z = graph.get_vertex_coords(v)
        if mask[int(round(x)), int(round(y)), int(round(z))]:
            graph.remove_vertex(v)
    print('\tBuilding geometry...')
    graph.build_vertex_geometry()

    print('\tCLAHE on filed_value_inv property...')
    graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.clahe_field_value(max_geo_dist=50, N=256, clip_f=100., s_max=4.)

    print('\tComputing vertices and edges properties...')
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()
    graph.add_prop_inv(prop_topo, edg=True)
    graph.compute_edge_affinity()

    print('\tApplying general thresholds...')
    if ang_rot is not None:
        print('\tDeleting edges in MW area...')
        graph.filter_mw_edges(ang_rot, ang_tilt)

    print('\tComputing graph global statistics (before simplification)...')
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_den)
    print('\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3')
    print('\t\t-Edge density: ' + str(round(nev,5)) + ' nm^3')
    print('\t\t-Edge/Vertex ratio: ' + str(round(nepv,5)))

    print('\tGraph density simplification for vertices...')
    if prop_topo != ps.globals.STR_FIELD_VALUE:
        print('\t\tProperty used: ' + prop_topo)
        graph.set_pair_prop(prop_topo)
    try:
        graph.graph_density_simp_ref(mask=np.asarray(mask_den, dtype=int), v_den=v_den,
                                     v_prop=v_prop, v_mode=v_mode)
    except ps.pexceptions.PySegInputWarning as e:
        print('WARNING: graph density simplification failed:')
        print('\t-' + e.get_message())

    print('\tGraph density simplification for edges in the membrane...')
    mask_mb = (seg == 1) * (~mask)
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_mb)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_mb, dtype=int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)

    print('\tGraph density simplification for edges in the PSD...')
    mask_psd = (seg == 2) * (~mask)
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_psd)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_psd, dtype=int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print('\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv))

    print('\tGraph density simplification for edges in the AZ...')
    mask_az = (seg == 3) * (~mask)
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_az)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_az, dtype=int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print('\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv))

    print('\tComputing graph global statistics (after simplification)...')
    nvv, _, _ = graph.compute_global_stat(mask=mask_den)
    _, nev_mb, nepv_mb = graph.compute_global_stat(mask=mask_mb)
    _, nev_psd, nepv_psd = graph.compute_global_stat(mask=mask_psd)
    _, nev_az, nepv_az = graph.compute_global_stat(mask=mask_az)
    print('\t\t-Vertex density (membrane): ' + str(round(nvv,5)) + ' nm^3')
    print('\t\t-Edge density (membrane):' + str(round(nev_mb,5)) + ' nm^3')
    print('\t\t-Edge density (PSD):' + str(round(nev_psd,5)) + ' nm^3')
    print('\t\t-Edge density (AZ):' + str(round(nev_az,5)) + ' nm^3')
    print('\t\t-Edge/Vertex ratio (membrane): ' + str(round(nepv_mb,5)))
    print('\t\t-Edge/Vertex ratio (PSD): ' + str(round(nepv_psd,5)))
    print('\t\t-Edge/Vertex ratio (AZ): ' + str(round(nepv_az,5)))

    print('\tComputing graph properties (2)...')
    graph.compute_mb_geo(update=True)
    graph.compute_mb_eu_dst()
    graph.compute_edge_curvatures()
    graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()
    graph.compute_edge_affinity()

    # print '\tCLAHE within membrane...'
    # graph.clahe_field_value(max_geo_dist=50, N=256, clip_f=100., s_max=4.)

    print('\tSaving intermediate graphs...')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + stem + '_edges_2.vtp')
    # ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
    #                         output_dir + '/' + stem + '_sch.vtp')

    out_pkl = output_dir + '/' + stem_pkl + '.pkl'
    print('\tPickling the graph as: ' + out_pkl)
    graph.pickle(out_pkl)
    star.set_element('_psGhMCFPickle', row, out_pkl)

out_star = output_dir + '/' + os.path.splitext(os.path.split(in_star)[1])[0] + '_mb_graph.star'
print('\tStoring output STAR file in: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')