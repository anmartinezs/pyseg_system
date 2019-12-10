"""

    Script for extracting an analyzing the filaments attached to a membrane from several input datasets
    DEPRECATED: USE mb_graph_batch.py instead

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
import operator
import numpy as np
from pyseg.mb import MbFilaments
try:
    import cPickle as pickle
except:
    import pickle

########## Global variables

STR_SEG = 'mb_seg'
STR_DST = 'dst'

# Membrane segmentation: 1-mb, 2-cito, 3-ext
SEG_MB = 1
SEG_MB_IN = 2
SEG_MB_OUT = 3

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/workspace/disperse/data/psd_an1/in/zd/bin4'

# Original density map
input_tomo_l = (ROOT_PATH+'/syn_14_7_bin4_sirt_rot_crop2.fits',
                ROOT_PATH+'/syn_14_9_bin4_sirt_rot_crop2.fits',
                ROOT_PATH+'/syn_14_13_bin4_sirt_rot_crop2.fits',
                ROOT_PATH+'/syn_14_14_bin4_sirt_rot_crop2.fits',
                ROOT_PATH+'/syn_14_15_bin4_sirt_rot_crop2.fits'
                )
# Membrane segmentation: 1-mb, 2-cito, 3-ext
input_seg_l = (ROOT_PATH+'/syn_14_7_bin4_crop2_pst_seg.fits',
               ROOT_PATH+'/syn_14_9_bin4_crop2_pst_seg.fits',
               ROOT_PATH+'/syn_14_13_bin4_crop2_pst_seg.fits',
               ROOT_PATH+'/syn_14_14_bin4_crop2_pst_seg.fits',
               ROOT_PATH+'/syn_14_15_bin4_crop2_pst_seg.fits',
               )

####### Output data

output_dir = '/home/martinez/workspace/disperse/data/psd_an1/zd/pst/mb_graph_bin4'

####### GraphMCF

disp_nsig = (0.1,
	        0.1,
	        0.1,
	        0.15,
	        0.15,)
ang_rot = None # -96.9174
ang_tilt = None # 60
s_sig = 0.5
nstd = 3
smooth = 3
res = 1.368 # nm/pix
mb_dst_off = 0. # nm
DILATE_NITER = 2 # pix

####### General thresholds

fness_th = None # 0.62
len_th_per = 60 # %

######## Filament thresholds

build_net = False
# Pre-thresholds
nrad = 5. # nm
max_len = 60 # nm

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print 'Extracting GraphMCF and NetFilament objects from tomograms'
print 'DEPRECATED: USE mb_graph_batch.py instead'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tDisPerSe persistence thresholds (nsig): ' + str(disp_nsig)
if ang_rot is not None:
    print 'Missing wedge edge compensation (rot, tilt): (' + str(ang_rot) + ', ' + str(ang_tilt) + ')'
print '\tSigma for gaussian pre-processing: ' + str(s_sig)
print '\tSigma for contrast enhancement: ' + str(nstd)
print '\tSkeleton smoothing factor: ' + str(smooth)
print '\tData resolution: ' + str(res) + ' nm/pixel'
print '\tMask offset: ' + str(mb_dst_off) + ' nm'
print 'Graph thresholds:'
if fness_th is not None:
    print '\tFilamentness down threhold: ' + str(fness_th)
if len_th_per is not None:
    print '\tPercentile for edge length: ' + str(len_th_per) + '%'
if build_net is not None:
    print 'Membrane filament tracking activated with thresholds.'
    print '\tNeighbourhood radius for computing the normals: ' + str(nrad) + ' nm'
    print '\tMaximum length: ' + str(max_len) + ' nm'
print ''

# Loop for processing the input data
print 'Running main loop: '
for (input_tomo, input_seg, nsig) in zip(input_tomo_l, input_seg_l, disp_nsig):

    print '\tComputing paths for' + input_tomo + ' ...'
    path, stem_tomo = os.path.split(input_tomo)
    stem_pkl, _ = os.path.splitext(stem_tomo)
    input_file = output_dir + '/' + stem_pkl + '_g' + str(s_sig) + '.fits'
    _, stem = os.path.split(input_file)
    stem, _ = os.path.splitext(stem)

    print '\tLoading input data: ' + stem_tomo
    tomo = ps.disperse_io.load_tomo(input_tomo)
    segh = ps.disperse_io.load_tomo(input_seg)

    print '\tComputing distance, mask and segmentation tomograms...'
    tomod = ps.disperse_io.seg_dist_trans(segh == SEG_MB) * res
    maskh = np.ones(shape=segh.shape, dtype=np.int)
    maskh[DILATE_NITER:-DILATE_NITER, DILATE_NITER:-DILATE_NITER, DILATE_NITER: -DILATE_NITER] = 0
    mask = np.asarray(tomod > (max_len + mb_dst_off + 2*DILATE_NITER*res), dtype=np.int)
    maskh += mask
    mask = np.asarray(maskh > 0, dtype=np.float)
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

    print '\tSmoothing input tomogram...'
    density = sp.ndimage.filters.gaussian_filter(tomo, s_sig)
    density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1)
    ps.disperse_io.save_numpy(tomo, output_dir + '/' + stem_pkl + '.vti')
    ps.disperse_io.save_numpy(density.transpose(), input_file)

    print '\tInitializing DisPerSeg...'
    work_dir = output_dir + '/disperse'
    disperse = ps.disperse_io.DisPerSe(input_file, work_dir)
    disperse.clean_work_dir()
    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    # Down skeleton
    disperse.set_dump_arcs(-1)
    disperse.set_nsig_cut(nsig)
    disperse.set_mask(input_msk)
    disperse.set_smooth(smooth)

    print '\tRunning DisPerSe...'
    disperse.mse(no_cut=False, inv=False)
    skel = disperse.get_skel()
    manifolds = disperse.get_manifolds(no_cut=False, inv=False)

    # Build the GraphMCF for the membrane
    print '\tBuilding MCF graph...'
    graph = ps.graph.GraphMCF(skel, manifolds, density)
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

    print '\tComputing graph properties...'
    graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()

    print '\tAdding segmentation...'
    graph.add_scalar_field_nn(seg, STR_SEG)
    graph.add_scalar_field(tomod, STR_DST)

    print '\tSaving intermediate graphs...'
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + stem + '_edges_2.vtp')
    ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                            output_dir + '/' + stem + '_sch.vtp')

    print '\tApplying general thresholds...'
    if ang_rot is not None:
        print '\tDeleting edges in MW area...'
        graph.filter_mw_edges(ang_rot, ang_tilt)
    if len_th_per is not None:
        length_th = graph.find_per_th(ps.globals.SGT_EDGE_LENGTH, vertex=False, per_ct=len_th_per)
        print '\tUp-thresholding edges with edge length: ' + str(length_th) + ' nm'
        graph.threshold_edges(ps.globals.SGT_EDGE_LENGTH, length_th, operator.gt)
    if fness_th is not None:
        graph.threshold_edges(ps.globals.STR_EDGE_FNESS, fness_th, operator.lt)
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + stem + '_edges_3.vtp')
    
    print '\tPickling the graph as: ' + stem_pkl + '.pkl'
    ps.disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')
    graph.pickle(output_dir + '/' + stem_pkl + '.pkl')
    
    if build_net:

      print '\tBuilding the filaments...'
      net_fils = MbFilaments(graph, seg, nrad=nrad, max_length=max_len)

      print '\tClustering...'
      net_fils.cont_clusters(SEG_MB_IN, approx=True)
      net_fils.cont_clusters(SEG_MB_OUT, approx=True)
      net_fils.cont_clusters_eu(SEG_MB_IN)
      net_fils.cont_clusters_eu(SEG_MB_OUT)

      print '\tStoring result network as: ' + stem + '_net.pkl'
      ps.disperse_io.save_vtp(net_fils.get_cont_vtp(force_update=True),
			      output_dir + '/' + stem + '_mb_fil_cont.vtp')
      ps.disperse_io.save_vtp(net_fils.get_sch_vtp(force_update=False),
			      output_dir + '/' + stem + '_mb_fil_sch.vtp')
      ps.disperse_io.save_vtp(net_fils.get_path_vtp(force_update=False),
			      output_dir + '/' + stem + '_mb_fil_path.vtp')
      net_fils.pickle(output_dir + '/' + stem + '_net.pkl')

print 'Terminated. (' + time.strftime("%c") + ')'
