"""

    Script for extracting the graphs for analyzing quantitatively a synapse
    (or any other junction between two membranes)

    Input:  - A STAR file with segmentations

    Output: - A graph

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import os
import time
import scipy
import pyseg as ps
from pyseg.mb.variables import *
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

SEG_TAG = '_seg'

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/pool/pub/4Antonio/spacer'

# Input STAR file with segmentations
in_star = ROOT_PATH + '/seg/test_hold.star'

####### Output data

# output_dir = ROOT_PATH+'/ex/syn/graphs_3'
output_dir = ROOT_PATH+'/graphs'

####### GraphMCF

s_sig = 0.5
csig = 0.1
ang_rot = None # -96.9174
ang_tilt = None # 60
nstd = 3
smooth = 3
res = 1.048 # nm/pix
MSK_OFF = 2 # pix

######## Masking thresholds
max_len_psd = 10 # nm
max_len_az = 10 # nm

####### Graph density thresholds

v_den = 0.006 # 0.003 # nm^3
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
print('Extracting SynGraphMCF from membrane pairs.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tSTAR file with the segmentations: ' + str(in_star))
print('\tDisPerSe persistence threshold (csig): ' + str(csig))
if ang_rot is not None:
    print('Missing wedge edge compensation (rot, tilt): (' + str(ang_rot) + ', ' + str(ang_tilt) + ')')
print('\tSigma for gaussian pre-processing: ' + str(s_sig))
print('\tSigma for contrast enhancement: ' + str(nstd))
print('\tSkeleton smoothing factor: ' + str(smooth))
print('\tData resolution: ' + str(res) + ' nm/pixel')
print('\tOutput directory: ' + output_dir)
print('Graph density thresholds:')
if v_prop is None:
    print('\tTarget vertex density (membranes) ' + str(v_den) + ' vertex/nm^3 for topological simplification')
else:
    print('\tTarget vertex density (membranes) ' + str(v_den) + ' vertex/nm^3 for property ' + v_prop + ' with mode ' + v_mode)
print('\tTarget edge/vertex ratio (non membrane) ' + str(ve_ratio) + ' for property ' + e_prop + ' with mode ' + e_mode)
print('')

print('Paring input star file...')
star = ps.sub.Star()
star.load(in_star)
in_tomor_l, in_seg_l = star.get_column_data('_rlnMicrographName'), star.get_column_data('_psSegImage')
star.add_column('_psGhMCFPickle')

# Loop for processing the input data
print('Running main loop: ')
for (row, in_tomor, in_seg) in zip(list(range(star.get_ncols())), in_tomor_l, in_seg_l):

    seg_tag_pos = in_seg.find(SEG_TAG)
    if seg_tag_pos < 0:
        print('WARNING: no cropped sub-volume found for segmentation: ' + in_seg)
    in_tomo = in_seg[:seg_tag_pos] + in_seg[seg_tag_pos+len(SEG_TAG):]
    print('\tSub-volume to process found: ' + in_tomo)
    print('\tComputing paths for ' + in_tomo + ' ...')
    f_path, f_fname = os.path.split(in_tomo)
    f_stem_pkl, f_ext = os.path.splitext(f_fname)
    input_file = output_dir + '/' + f_stem_pkl + '_g' + str(s_sig) + '.fits'
    _, f_stem = os.path.split(input_file)
    f_stem, _ = os.path.splitext(f_stem)

    print('\tLoading input data: ' + f_fname)
    tomo = ps.disperse_io.load_tomo(in_tomo)
    seg = ps.disperse_io.load_tomo(in_seg)

    print('\tComputing masks and segmentation tomograms...')
    tomod_pst = (ps.disperse_io.seg_dist_trans(seg==SYN_PST_LBL)*res) < (max_len_psd + 2*MSK_OFF*res)
    tomod_pre = (ps.disperse_io.seg_dist_trans(seg==SYN_PRE_LBL)*res) < (max_len_az + 2*MSK_OFF*res)
    tomoh = np.zeros(shape=seg.shape, dtype=bool)
    tomoh[MSK_OFF:-MSK_OFF, MSK_OFF:-MSK_OFF, MSK_OFF:-MSK_OFF] = True
    mask = ((tomoh & (tomod_pst | tomod_pre)) == False).astype(float)
    input_msk = output_dir + '/' + f_stem_pkl + '_mask.fits'
    ps.disperse_io.save_numpy(mask.transpose(), input_msk)
    mask = mask == False
    mask_den = ((seg == SYN_PST_LBL) | (seg == SYN_PRE_LBL)) & mask

    print('\tSmoothing input tomogram (s=' + str(s_sig) + ')...')
    density = scipy.ndimage.filters.gaussian_filter(tomo, s_sig)
    density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1)
    ps.disperse_io.save_numpy(tomo, output_dir + '/' + f_stem_pkl + '.vti')
    ps.disperse_io.save_numpy(density.transpose(), input_file)
    ps.disperse_io.save_numpy(density, output_dir + '/' + f_stem + '.vti')

    print('\tInitializing DisPerSe...')
    work_dir = output_dir + '/disperse'
    disperse = ps.disperse_io.DisPerSe(input_file, work_dir)
    try:
        disperse.clean_work_dir()
    except ps.pexceptions.PySegInputWarning as e:
        print(e.get_message())

    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    # Down skeleton
    disperse.set_dump_arcs(-1)
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
    graph = ps.mb.SynGraphMCF(skel, manifolds, density, seg)
    graph.set_resolution(res)
    graph.filter_self_edges()
    graph.filter_repeated_edges()

    print('\tFiltering nodes close to mask border...')
    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=MSK_OFF)
    for v in graph.get_vertices_list():
        x, y, z = graph.get_vertex_coords(v)
        if not mask[int(round(x)), int(round(y)), int(round(z))]:
            graph.remove_vertex(v)
    print('\tBuilding geometry...')
    graph.build_vertex_geometry()

    print('\tCLAHE on filed_value_inv property...')
    graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.clahe_field_value(max_geo_dist=50, N=256, clip_f=100., s_max=4.)

    # ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
    #                         output_dir + '/hold.vtp')

    print('\tComputing vertices and edges properties...')
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()
    if prop_topo is None:
        graph.add_prop_inv(ps.globals.STR_FIELD_VALUE, edg=True)
    else:
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

    print('\tGraph density simplification for edges in the pst membrane...')
    mask_pst = (seg == SYN_PST_LBL) & mask
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_pst)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_pst, dtype=int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print('\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv))

    print('\tGraph density simplification for edges in the pre membrane...')
    mask_pre = (seg == SYN_PRE_LBL) & mask
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_pre)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_pre, dtype=int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print('\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv))

    print('\tGraph density simplification for edges in the PSD...')
    mask_psd = (seg == SYN_PSD_LBL) & mask
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_psd)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_psd, dtype=int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print('\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv))

    print('\tGraph density simplification for edges in the AZ...')
    mask_az = (seg == SYN_AZ_LBL) & mask
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_az)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_az, dtype=int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print('\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv))

    print('\tGraph density simplification for edges in the cleft...')
    mask_clf = (seg == SYN_CLF_LBL) & mask
    nvv, nev, nepv = graph.compute_global_stat(mask=mask_clf)
    if nepv > ve_ratio:
        e_den = nvv * ve_ratio
        hold_e_prop = e_prop
        graph.graph_density_simp_ref(mask=np.asarray(mask_clf, dtype=int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print('\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv))

    print('\tComputing graph global statistics (after simplification)...')
    nvv, _, _ = graph.compute_global_stat(mask=mask_den)
    _, nev_pst, nepv_pst = graph.compute_global_stat(mask=mask_pst)
    _, nev_pre, nepv_pre = graph.compute_global_stat(mask=mask_pre)
    _, nev_psd, nepv_psd = graph.compute_global_stat(mask=mask_psd)
    _, nev_az, nepv_az = graph.compute_global_stat(mask=mask_az)
    _, nev_clf, nepv_clf = graph.compute_global_stat(mask=mask_clf)
    print('\t\t-Vertex density (membranes): ' + str(round(nvv,5)) + ' nm^3')
    print('\t\t-Edge density (PST):' + str(round(nev_pst,5)) + ' nm^3')
    print('\t\t-Edge density (PRE):' + str(round(nev_pre,5)) + ' nm^3')
    print('\t\t-Edge density (PSD):' + str(round(nev_psd,5)) + ' nm^3')
    print('\t\t-Edge density (AZ):' + str(round(nev_az,5)) + ' nm^3')
    print('\t\t-Edge density (CLFT):' + str(round(nev_clf,5)) + ' nm^3')
    print('\t\t-Edge/Vertex ratio (PST): ' + str(round(nepv_pst,5)))
    print('\t\t-Edge/Vertex ratio (PRE): ' + str(round(nepv_pre,5)))
    print('\t\t-Edge/Vertex ratio (PSD): ' + str(round(nepv_psd,5)))
    print('\t\t-Edge/Vertex ratio (AZ): ' + str(round(nepv_az,5)))
    print('\t\t-Edge/Vertex ratio (CLFT): ' + str(round(nepv_az,5)))

    print('\tComputing graph properties (2)...')
    graph.compute_mb_geo()
    graph.compute_mb_eu_dst()
    graph.compute_edge_curvatures()
    graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()
    graph.compute_edge_affinity()


    print('\tSaving intermediate graphs...')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + f_stem + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + f_stem + '_edges_2.vtp')
    # ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
    #                         output_dir + '/' + f_stem + '_sch.vtp')

    out_pkl = output_dir + '/' + f_stem_pkl + '.pkl'
    print('\tPickling the graph as: ' + out_pkl)
    graph.pickle(out_pkl)
    star.set_element('_psGhMCFPickle', row, out_pkl)

out_star = output_dir + '/' + os.path.splitext(os.path.split(in_star)[1])[0] + '_mb_graph.star'
print('\tStoring output STAR file in: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')