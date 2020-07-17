"""

    Script for extracting an analyzing a SynGraphMCF from an oriented pairs of membranes (like a synapse)

    Input:  - A STAR file with a list of (sub-)tomograms to process:
	      	+ Density map tomogram
            	+ Segmentation tomogram
            - Graph input parameters

    Output: - A STAR file with the (sub-)tomograms and their correspoing graphs
              (MbGraphMCF object)
            - Additional files for visualization

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import time
import sys
import math
import pyseg as ps
import scipy as sp
import os
import numpy as np
import multiprocessing as mp

try:
    import cPickle as pickle
except:
    import pickle

########## Global variables

MB_LBL_1, MB_LBL_2 = 1, 2
EXT_LBL_1, EXT_LBL_2 = 3, 4
GAP_LBL, BG_LBL = 5, 0

########################################################################################
# INPUT PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/pool/pool-ruben/antonio/nuc_mito'  # Data path

# Input STAR file with segmentations
in_star = ROOT_PATH + '/pre/mbdo_nosplit/dmb_seg_oriented_pre.star'
npr = 1 # number of parallel processes

####### Output data

output_dir = ROOT_PATH + '/graphs/v2'

####### GraphMCF perameter

res = 1.408  # nm/pix
s_sig = 0.75  # 1.5
v_den = 0.0035  # 0.007 # 0.0025 # nm^3
ve_ratio = 2 # 4
max_len = 10  # 15 # 30 # nm

####### Advanced parameters

# nsig = 0.01
csig = 0.01
ang_rot = None
ang_tilt = None
nstd = 5 # 3 # 10
smooth = 3
mb_dst_off = 5  # nm
DILATE_NITER = 2  # pix
do_clahe = False  # True

####### Graph density thresholds

v_prop = None  # ps.globals.STR_FIELD_VALUE # In None topological simplification
e_prop = ps.globals.STR_FIELD_VALUE  # ps.globals.STR_FIELD_VALUE_EQ # ps.globals.STR_VERT_DST
v_mode = None  # 'low'
e_mode = 'low'
prop_topo = ps.globals.STR_FIELD_VALUE  # ps.globals.STR_FIELD_VALUE_EQ # None is ps.globals.STR_FIELD_VALUE

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print 'Extracting GraphMCF and NetFilament objects from tomograms'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
# print '\tDisPerSe persistence threshold (nsig): ' + str(nsig)
print '\tSTAR file with the segmentations: ' + str(in_star)
print '\tNumber of parallel processes: ' + str(npr)
print '\tDisPerSe persistence threshold (csig): ' + str(csig)
if ang_rot is not None:
    print 'Missing wedge edge compensation (rot, tilt): (' + str(ang_rot) + ', ' + str(ang_tilt) + ')'
print '\tSigma for gaussian pre-processing: ' + str(s_sig)
print '\tSigma for contrast enhancement: ' + str(nstd)
print '\tSkeleton smoothing factor: ' + str(smooth)
print '\tData resolution: ' + str(res) + ' nm/pixel'
print '\tMask offset: ' + str(mb_dst_off) + ' nm'
print '\tOutput directory: ' + output_dir
print 'Graph density thresholds:'
if v_prop is None:
    print '\tTarget vertex density (membrane) ' + str(v_den) + ' vertex/nm^3 for topological simplification'
else:
    print '\tTarget vertex density (membrane) ' + str(
        v_den) + ' vertex/nm^3 for property ' + v_prop + ' with mode ' + v_mode
print '\tTarget edge/vertex ratio (non membrane) ' + str(ve_ratio) + ' for property ' + e_prop + ' with mode ' + e_mode
if do_clahe:
    print '\t-Computing CLAHE.'
print ''

print 'Paring input star file...'
star = ps.sub.Star()
star.load(in_star)
in_seg_l = star.get_column_data('_psSegImage')
in_tomo_l = star.get_column_data('_rlnImageName')
star.add_column('_psGhMCFPickle')

### Parallel worker
def pr_worker(pr_id, ids, q_pkls):

    pkls_dic = dict()
    for row in ids:

        input_seg, input_tomo = in_seg_l[row], in_tomo_l[row]

        print '\tP[' + str(pr_id) + '] Sub-volume to process found: ' + input_tomo
        print '\tP[' + str(pr_id) + '] Computing paths for ' + input_tomo + ' ...'
        path, stem_tomo = os.path.split(input_tomo)
        stem_pkl, _ = os.path.splitext(stem_tomo)
        input_file = output_dir + '/' + stem_pkl + '_g' + str(s_sig) + '.fits'
        _, stem = os.path.split(input_file)
        stem, _ = os.path.splitext(stem)

        print '\tP[' + str(pr_id) + '] Loading input data: ' + stem_tomo
        tomo = ps.disperse_io.load_tomo(input_tomo).astype(np.float32)
        seg = ps.disperse_io.load_tomo(input_seg)

        print '\tP[' + str(pr_id) + '] Computing masks and segmentation tomograms...'
        tomoh = np.zeros(shape=seg.shape, dtype=np.bool)
        mb_dst_off_v = int(math.ceil(mb_dst_off * res))
        tomoh[mb_dst_off_v:-mb_dst_off_v, mb_dst_off_v:-mb_dst_off_v, mb_dst_off_v:-mb_dst_off_v] = True
        mask = ((tomoh & (seg != BG_LBL)) == False).astype(np.float)
        input_msk = output_dir + '/' + stem_pkl + '_mask.fits'
        ps.disperse_io.save_numpy(mask.transpose(), input_msk)
        mask = mask == False
        mask_den = ((seg == MB_LBL_1) | (seg == MB_LBL_2)) & mask

        print '\tP[' + str(pr_id) + '] Smoothing input tomogram (s=' + str(s_sig) + ')...'
        density = sp.ndimage.filters.gaussian_filter(tomo, s_sig)
        density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1, mask=mask)
        ps.disperse_io.save_numpy(tomo, output_dir + '/' + stem_pkl + '.vti')
        ps.disperse_io.save_numpy(density.transpose(), input_file)
        ps.disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')

        print '\tP[' + str(pr_id) + '] Initializing DisPerSe...'
        work_dir = output_dir + '/disperse_pr_' + str(pr_id)
        disperse = ps.disperse_io.DisPerSe(input_file, work_dir)
        try:
            disperse.clean_work_dir()
        # except ps.pexceptions.PySegInputWarning as e:
        #     print e.get_message()
        except Warning:
            print 'Jol!!!'

        # Manifolds for descending fields with the inverted image
        disperse.set_manifolds('J0a')
        # Down skeleton
        disperse.set_dump_arcs(-1)
        # disperse.set_nsig_cut(nsig)
        rcut = round(density[mask_den].std() * csig, 4)
        print '\tP[' + str(pr_id) + '] Persistence cut thereshold set to: ' + str(rcut) + ' grey level'
        disperse.set_cut(rcut)
        disperse.set_mask(input_msk)
        disperse.set_smooth(smooth)

        print '\tP[' + str(pr_id) + '] Running DisPerSe...'
        disperse.mse(no_cut=False, inv=False)
        skel = disperse.get_skel()
        manifolds = disperse.get_manifolds(no_cut=False, inv=False)

        # Build the GraphMCF for the membrane
        print '\tP[' + str(pr_id) + '] Building MCF graph for a pair of oriented membranes...'
        # graph = ps.mb.MbGraphMCF(skel, manifolds, density, seg)
        graph = ps.mb.SynGraphMCF(skel, manifolds, density, seg)
        graph.set_resolution(res)
        graph.build_from_skel(basic_props=False)
        graph.filter_self_edges()
        graph.filter_repeated_edges()

        print '\tP[' + str(pr_id) + '] Filtering nodes close to mask border...'
        mask = sp.ndimage.morphology.binary_dilation(mask, iterations=DILATE_NITER)
        for v in graph.get_vertices_list():
            x, y, z = graph.get_vertex_coords(v)
            if not mask[int(round(x)), int(round(y)), int(round(z))]:
                graph.remove_vertex(v)
        print '\tP[' + str(pr_id) + '] Building geometry...'
        graph.build_vertex_geometry()

        if do_clahe:
            print '\tP[' + str(pr_id) + '] CLAHE on filed_value_inv property...'
            graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
            graph.clahe_field_value(max_geo_dist=50, N=256, clip_f=100., s_max=4.)

        print '\tP[' + str(pr_id) + '] Computing vertices and edges properties...'
        graph.compute_vertices_dst()
        graph.compute_edge_filamentness()
        graph.add_prop_inv(prop_topo, edg=True)
        graph.compute_edge_affinity()

        print '\tP[' + str(pr_id) + '] Applying general thresholds...'
        if ang_rot is not None:
            print '\tDeleting edges in MW area...'
            graph.filter_mw_edges(ang_rot, ang_tilt)

        print '\tP[' + str(pr_id) + '] Computing graph global statistics (before simplification)...'
        nvv, nev, nepv = graph.compute_global_stat(mask=mask_den)
        print '\t\t-P[' + str(pr_id) + '] Vertex density: ' + str(round(nvv, 5)) + ' nm^3'
        print '\t\t-P[' + str(pr_id) + '] Edge density: ' + str(round(nev, 5)) + ' nm^3'
        print '\t\t-P[' + str(pr_id) + '] Edge/Vertex ratio: ' + str(round(nepv, 5))

        print '\tP[' + str(pr_id) + '] Graph density simplification for vertices...'
        if prop_topo != ps.globals.STR_FIELD_VALUE:
            print '\t\tProperty used: ' + prop_topo
            graph.set_pair_prop(prop_topo)
        try:
            graph.graph_density_simp_ref(mask=np.asarray(mask_den, dtype=np.int), v_den=v_den,
                                         v_prop=v_prop, v_mode=v_mode)
        except ps.pexceptions.PySegInputWarning as e:
            print 'P[' + str(pr_id) + '] WARNING: graph density simplification failed:'
            print '\t-' + e.get_message()

        print '\tGraph density simplification for edges in membrane 1...'
        mask_pst = (seg == MB_LBL_1) & mask
        nvv, nev, nepv = graph.compute_global_stat(mask=mask_pst)
        if nepv > ve_ratio:
            e_den = nvv * ve_ratio
            hold_e_prop = e_prop
            graph.graph_density_simp_ref(mask=np.asarray(mask_pst, dtype=np.int), e_den=e_den,
                                         e_prop=hold_e_prop, e_mode=e_mode, fit=True)
        else:
            print '\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv)

        print '\tGraph density simplification for edges in the membrane 2...'
        mask_pre = (seg == MB_LBL_2) & mask
        nvv, nev, nepv = graph.compute_global_stat(mask=mask_pre)
        if nepv > ve_ratio:
            e_den = nvv * ve_ratio
            hold_e_prop = e_prop
            graph.graph_density_simp_ref(mask=np.asarray(mask_pre, dtype=np.int), e_den=e_den,
                                         e_prop=hold_e_prop, e_mode=e_mode, fit=True)
        else:
            print '\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv)

        print '\tGraph density simplification for edges in the exterior 1...'
        mask_psd = (seg == EXT_LBL_1) & mask
        nvv, nev, nepv = graph.compute_global_stat(mask=mask_psd)
        if nepv > ve_ratio:
            e_den = nvv * ve_ratio
            hold_e_prop = e_prop
            graph.graph_density_simp_ref(mask=np.asarray(mask_psd, dtype=np.int), e_den=e_den,
                                         e_prop=hold_e_prop, e_mode=e_mode, fit=True)
        else:
            print '\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv)

        print '\tGraph density simplification for edges in the exterior 2...'
        mask_az = (seg == EXT_LBL_2) & mask
        nvv, nev, nepv = graph.compute_global_stat(mask=mask_az)
        if nepv > ve_ratio:
            e_den = nvv * ve_ratio
            hold_e_prop = e_prop
            graph.graph_density_simp_ref(mask=np.asarray(mask_az, dtype=np.int), e_den=e_den,
                                         e_prop=hold_e_prop, e_mode=e_mode, fit=True)
        else:
            print '\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv)

        print '\tGraph density simplification for edges in the gap...'
        mask_clf = (seg == GAP_LBL) & mask
        nvv, nev, nepv = graph.compute_global_stat(mask=mask_clf)
        if nepv > ve_ratio:
            e_den = nvv * ve_ratio
            hold_e_prop = e_prop
            graph.graph_density_simp_ref(mask=np.asarray(mask_clf, dtype=np.int), e_den=e_den,
                                         e_prop=hold_e_prop, e_mode=e_mode, fit=True)
        else:
            print '\tWARNING: demanded ratio ' + str(nepv) + ' could not be achieved (current is ' + str(nepv)

        print '\tComputing graph global statistics (after simplification)...'
        nvv, _, _ = graph.compute_global_stat(mask=mask_den)
        _, nev_pst, nepv_pst = graph.compute_global_stat(mask=mask_pst)
        _, nev_pre, nepv_pre = graph.compute_global_stat(mask=mask_pre)
        _, nev_psd, nepv_psd = graph.compute_global_stat(mask=mask_psd)
        _, nev_az, nepv_az = graph.compute_global_stat(mask=mask_az)
        _, nev_clf, nepv_clf = graph.compute_global_stat(mask=mask_clf)
        print '\t\t-Vertex density (membranes): ' + str(round(nvv, 5)) + ' nm^3'
        print '\t\t-Edge density (MB1):' + str(round(nev_pst, 5)) + ' nm^3'
        print '\t\t-Edge density (MB2):' + str(round(nev_pre, 5)) + ' nm^3'
        print '\t\t-Edge density (EXT1):' + str(round(nev_psd, 5)) + ' nm^3'
        print '\t\t-Edge density (EXT2):' + str(round(nev_az, 5)) + ' nm^3'
        print '\t\t-Edge density (GAP):' + str(round(nev_clf, 5)) + ' nm^3'
        print '\t\t-Edge/Vertex ratio (MB1): ' + str(round(nepv_pst, 5))
        print '\t\t-Edge/Vertex ratio (MB2): ' + str(round(nepv_pre, 5))
        print '\t\t-Edge/Vertex ratio (EXT1): ' + str(round(nepv_psd, 5))
        print '\t\t-Edge/Vertex ratio (EXT2): ' + str(round(nepv_az, 5))
        print '\t\t-Edge/Vertex ratio (GAP): ' + str(round(nepv_az, 5))

        print '\tComputing graph properties (2)...'
        graph.compute_mb_geo()
        graph.compute_mb_eu_dst()
        graph.compute_edge_curvatures()
        # graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
        graph.compute_vertices_dst()
        graph.compute_edge_filamentness()
        graph.compute_edge_affinity()

        print '\tSaving intermediate graphs...'
        ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                                output_dir + '/' + stem + '_edges.vtp')
        ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                                output_dir + '/' + stem + '_edges_2.vtp')
        # ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
        #                         output_dir + '/' + stem + '_sch.vtp')

        out_pkl = output_dir + '/' + stem_pkl + '.pkl'
        print '\tP[' + str(pr_id) + '] Pickling the graph as: ' + out_pkl
        graph.pickle(out_pkl)
        # star.set_element('_psGhMCFPickle', row, out_pkl)
        q_pkls.put((row, out_pkl))
        pkls_dic[row] = out_pkl

    sys.exit(pr_id)

# Loop for processing the input data
print 'Running main loop in parallel: '
q_pkls = mp.Queue()
processes, pr_results = dict(), dict()
spl_ids = np.array_split(range(star.get_nrows()), npr)
for pr_id in range(npr):
    pr = mp.Process(target=pr_worker, args=(pr_id, spl_ids[pr_id], q_pkls))
    pr.start()
    processes[pr_id] = pr
for pr_id, pr in zip(processes.iterkeys(), processes.itervalues()):
    pr.join()
    if pr_id != pr.exitcode:
        print 'ERROR: the process ' + str(pr_id) + ' ended unsuccessfully [' + str(pr.exitcode) + ']'
        print 'Unsuccessfully terminated. (' + time.strftime("%c") + ')'

count, n_rows = 0, star.get_nrows()
while count < n_rows:
    hold_out_pkl = q_pkls.get()
    star.set_element(key='_psGhMCFPickle', row=hold_out_pkl[0], val=hold_out_pkl[1])
    count += 1

out_star = output_dir + '/' + os.path.splitext(os.path.split(in_star)[1])[0] + '_mb_graph.star'
print '\tStoring output STAR file in: ' + out_star
star.store(out_star)

print 'Terminated. (' + time.strftime("%c") + ')'
