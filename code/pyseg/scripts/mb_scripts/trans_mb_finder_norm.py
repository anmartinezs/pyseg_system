"""

    Script for finding transmembrane high density structures (features)

    Input:  - Density map tomogram
            - Membrane segmentation tomogram
            - Segmentation parameters

    Output: - Transmembrane MCF graph
            - List (.csv) of feature coordinates and properties

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

import time
import pyseg as ps
import scipy as sp
import os
import sys
import numpy as np
import operator
from pyseg.sub import TomoPeaks
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

SVOL_OFF = 20
OR_CLOSEST_N, OR_NORMAL, OR_ROT_A = 'closest_neigh', 'normal', 'rotation_angs'
SWAP_XY = True # False

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/pool/pool-plitzko/Matthias/Tomography/4Antonio/'

# Original density map
input_tomo = ROOT_PATH + 'tomo9.mrc'

# Membrane segmentation: 1-membranes, 0-bg
input_seg = ROOT_PATH + 'tomo9_ref_mb_gf1.5.mrc'

# MembraT ne orientation: 1-membranes, 0-bg
input_omb = ROOT_PATH + 'tomo9_or_mb.mrc'

# List of key properties associated to features
prop_l = ('field_value_inv', 'ggf')

####### Output data

output_dir = ROOT_PATH + '/results'
out_log = None # output_dir + '/out.txt' # None then output is redirected to stdout

####### GraphMCF

s_sig = 1.0 # Gaussian pre-filter on the whole tomogram
csig = 0.05
nstd = 3
smooth = 3
res = 1.048 # nm/pix

####### Graph density thresholds

v_num = None # 90000
e_num = None # 180000
v_den = 0.01 # nm^3 (only valid if v_num is None)
e_den = None # nm^3 (only valid if e_num is None)
v_prop = ps.globals.STR_FIELD_VALUE_INV
e_prop = ps.globals.SGT_EDGE_LENGTH
v_mode = 'high'
e_mode = 'low'

####### Scale suppression

sup_scale = 2.5 # nm if None no scale suppression active
sup_prop_v = ps.globals.STR_FIELD_VALUE_INV
sup_conn = False

####### CSV parameters

crop_off = (0, 0, 0)

###### GGF parameters (only applicable if 'ggf' is in prop_l)

ggf_sig = 6 # nm when ggf_prop_e = ps.globals.STR_VERT_DST
ggf_prop_v = None # ps.globals.STR_FIELD_VALUE_INV # None
ggf_prop_e = ps.globals.STR_VERT_DST
ggf_vinv = False
ggf_einv = False
ggf_vnorm = False
ggf_enorm = False
ggf_energy = False # True

######## Masking thresholds
mb_dst_off = 0 # nm

########################################################################################
# MAIN ROUTINE
########################################################################################

if out_log is not None:
    print('Redirecting output text to: ' + out_log)
    sys.stdout = open(out_log, 'w')

# Print initial message
print('Extracting transmembrane features.')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
print('\tInput tomogram: ' + input_tomo)
print('\tInput membrane segmentation: ' + input_seg)
print('\tInput membrane orientation segmentation: ' + input_omb)
print('\tDisPerSe persistence threshold (csig): ' + str(csig))
print('\tSigma for gaussian pre-processing: ' + str(s_sig))
print('\tSigma for contrast enhancement: ' + str(nstd))
print('\tSkeleton smoothing factor: ' + str(smooth))
print('\tData resolution: ' + str(res) + ' nm/pixel')
print('\tOutput directory: ' + output_dir)
print('Graph density thresholds:')
if v_prop is None:
    if v_num is None:
        print('\tTarget vertex density (features) ' + str(v_den) + \
              ' vertex/nm^3 for topological simplification')
    else:
        print('\tTarget number of vertices (features) ' + str(v_num) + \
              ' vertices for topological simplification')
else:
    if v_num is None:
        print('\tTarget vertex density ' + str(v_den) + ' vertex/nm^3 for property ' + v_prop + \
              ' with mode ' + v_mode)
    else:
        print('\tTarget vertex density ' + str(v_num) + ' vertices for property ' + v_prop + \
              ' with mode ' + v_mode)
if e_num is None:
    print('\tTarget edge density ' + str(e_den) + ' edge/nm^3 for property ' + e_prop + \
          ' with mode ' + e_mode)
else:
    print('\tTarget edge density ' + str(e_num) + ' edges for property ' + e_prop + \
          ' with mode ' + e_mode)
if sup_scale is not None:
    print('Scale suppression:')
    print('\t-Scale ' + str(sup_scale) + ' nm')
    print('\t-Vertex sorting property ' + sup_prop_v)
    if not sup_conn:
        print('IMPORTANT: scale suppression without edge connectivity, this may spoil edge information!')
print('')

print('\tComputing paths for ' + input_tomo + ' ...')
path, stem_tomo = os.path.split(input_tomo)
stem_pkl, _ = os.path.splitext(stem_tomo)
input_file = output_dir + '/' + stem_pkl + '_g' + str(s_sig) + '.fits'
_, stem = os.path.split(input_file)
stem, _ = os.path.splitext(stem)
output_coords = output_dir + '/' + stem_pkl + '_peaks.coords'

print('\tLoading input data: ' + stem_tomo)
tomo = ps.disperse_io.load_tomo(input_tomo).astype(np.float32)
seg = ps.disperse_io.load_tomo(input_seg).astype(np.int16)
omb = ps.disperse_io.load_tomo(input_omb).astype(np.int16)
seg_fg = seg > 0
lbls = list(set(seg[seg_fg]))
tot_vol = float(seg_fg.sum())
del seg_fg
if len(lbls) == 0:
    print('ERROR: no segmented membrane to process found.')
    print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
    sys.exit(-1)
print('\t\t-Labels found ' + str(lbls))

print('\tLoop for processing labelled membranes...')
cont = 0
unproc_l = list()
for lbl in lbls:

    sys.stdout.flush()

    try:

        print('\tProcessing membranes with label ' + str(lbl))
        p_vol = float((seg == lbl).sum()) / tot_vol
        if v_num is not None:
            sv_num = int(np.ceil(v_num * p_vol))
            print('\t\tSubvolume number of target vertices: ' + str(sv_num))
        else:
            sv_num = None
        if e_num is not None:
            se_num = int(np.ceil(e_num * p_vol))
            print('\t\tSubvolume number of target edges: ' + str(se_num))
        else:
            se_num = None
        print('\t\tCropping subvolume for membrane with label ' + str(lbl) + \
              ' (Volume=' + str(round(p_vol*100, 2)) + '%)')
        stomo, sseg, scrop_off = ps.disperse_io.crop_lbl_tomo(tomo, seg, lbl, SVOL_OFF)
        somb = ps.disperse_io.crop_lbl_tomo(omb, seg, lbl, SVOL_OFF)[0]
        print('\t\t\t-Local offset: ' + str(scrop_off))
        gcrop_off = tuple(np.asarray(crop_off) + np.asarray(scrop_off))
        print('\t\t\t-Global offset: ' + str(gcrop_off))

        print('\t\tComputing mask...')
        tomod = ps.disperse_io.seg_dist_trans(sseg == lbl) * res
        mask = np.asarray(tomod <= (mb_dst_off/res), dtype=bool)
        input_msk = output_dir + '/' + stem + '_lbl_' + str(lbl) + '_mask.fits'
        ps.disperse_io.save_numpy(np.asarray(~mask, dtype=float).transpose(), input_msk)
        tomoo = somb > 0
        ps.disperse_io.save_numpy(somb, output_dir+'/'+stem + '_lbl_' + str(lbl) + '_orient.vti')

        print('\t\tSmoothing input tomogram (s=' + str(s_sig) + ')...')
        if s_sig > 0:
            density = sp.ndimage.filters.gaussian_filter(stomo, s_sig)
        else:
            density = np.asarray(stomo, dtype=float)
        density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1, mask=mask)
        ps.disperse_io.save_numpy(density, output_dir + '/' + stem_pkl + '_lbl_' + str(lbl) + '.vti')
        ps.disperse_io.save_numpy(density.transpose(), input_file)

        print('\t\tInitializing DisPerSeg...')
        work_dir = output_dir + '/disperse'
        disperse = ps.disperse_io.DisPerSe(input_file, work_dir)
        disperse.clean_work_dir()
        disperse.set_manifolds('J0a')
        disperse.set_dump_arcs(-1)
        rcut = round(density[mask].std()*csig, 4)

        print('\t\tPersistence cut threshold set to: ' + str(rcut) + ' grey level')
        disperse.set_cut(rcut)
        disperse.set_mask(input_msk)
        disperse.set_smooth(smooth)

        print('\t\tRunning DisPerSe...')
        disperse.mse(no_cut=False, inv=False)
        skel = disperse.get_skel()
        manifolds = disperse.get_manifolds(no_cut=False, inv=False)

        # Build the GraphMCF for the membrane
        print('\t\tBuilding MCF graph...')
        graph = ps.graph.GraphMCF(skel, manifolds, density)
        graph.set_resolution(res)
        graph.build_from_skel(basic_props=False)
        graph.filter_self_edges()
        graph.filter_repeated_edges()

        print('\t\tBuilding geometry...')
        graph.build_vertex_geometry()

        print('\t\tComputing vertices and edges properties...')
        graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
        graph.compute_vertices_dst()
        graph.compute_edge_filamentness()
        graph.add_prop_inv(ps.globals.STR_FIELD_VALUE)
        if ps.globals.STR_GGF == v_prop:
            print('\t\tTopological simplification based on GGF, pre-computing GGF...')
            graph_gt = ps.graph.GraphGT(graph)
            graph_gt.ggf(ggf_sig, ggf_prop_v, ggf_prop_e, ggf_vinv, ggf_einv, ggf_vnorm, ggf_enorm, ggf_energy)
            graph_gt.add_prop_to_GraphMCF(graph, ps.globals.STR_GGF, up_index=True)

        print('\t\tComputing graph global statistics (before simplification)...')
        nvv, nev, nepv = graph.compute_global_stat(mask=mask)
        print('\t\t\t-Number of vertices: ' + str(len(graph.get_vertices_list())))
        print('\t\t\t-Number of edges: ' + str(len(graph.get_edges_list())))
        print('\t\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3')
        print('\t\t\t-Edge density: ' + str(round(nev,5)) + ' nm^3')
        print('\t\t\t-Edge/Vertex ratio: ' + str(round(nepv,5)))

        print('\t\tGraph density simplification...')
        try:
            hold_e_prop = e_prop
            graph.graph_density_simp(v_num=sv_num, e_num=se_num, v_den=v_den, e_den=e_den, v_prop=v_prop, e_prop=hold_e_prop,
                                     v_mode=v_mode, e_mode=e_mode, mask=mask)
        except ps.pexceptions.PySegInputWarning as e:
            print('WARNING: graph density simplification failed:')
            print('\t-' + e.get_message())

        print('\t\tComputing graph global statistics (after simplification)...')
        nvv, nev, nepv = graph.compute_global_stat(mask=mask)
        print('\t\t\t-Number of vertices: ' + str(len(graph.get_vertices_list())))
        print('\t\t\t-Number of edges: ' + str(len(graph.get_edges_list())))
        print('\t\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3')
        print('\t\t\t-Edge density: ' + str(round(nev,5)) + ' nm^3')
        print('\t\t\t-Edge/Vertex ratio: ' + str(round(nepv,5)))

        graph_gt = None
        if ps.globals.STR_GGF in prop_l:
            print('\t\tComputing final GGF...')
            graph_gt = ps.graph.GraphGT(graph)
            graph_gt.ggf(ggf_sig, ggf_prop_v, ggf_prop_e, ggf_vinv, ggf_einv, ggf_vnorm, ggf_enorm, ggf_energy)
            graph_gt.add_prop_to_GraphMCF(graph, ps.globals.STR_GGF, up_index=True)

        if sup_scale is not None:
            print('\t\tVertices scale suppression...')
            if graph_gt is None:
                graph_gt = ps.graph.GraphGT(graph)
            vids = graph_gt.vertex_scale_supression(sup_scale, sup_prop_v, sup_conn)
            graph.remove_vertices_list(vids)

    except Exception as e:
        print('WARNING: membranes with label ' + str(lbl) + ' could not be processed because of ' + str(e))
        unproc_l.append(lbl)
        continue


    print('\t\tCreating the peaks container...')
    v_ids = [vertex.get_id() for vertex in graph.get_vertices_list()]
    tomo_peaks = TomoPeaks(shape=tomo.shape, name=input_tomo)
    tomo_peaks.add_peaks(graph.get_vertices_coords(v_ids))
    for prop in prop_l:
        tomo_peaks.add_prop(prop, n_comp=1, vals=graph.get_prop_values(prop, v_ids))

    print('\t\tAnalyzing orientation...')
    tomo_peaks.seg_shortest_pt(tomoo, OR_CLOSEST_N)
    tomo_peaks.vect_2pts(OR_CLOSEST_N, ps.sub.PK_COORDS, OR_NORMAL)
    tomo_peaks.vect_rotation_zrelion(key_v=OR_NORMAL, key_r=OR_ROT_A)

    print('\t\tSaving intermediate graphs...')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_lbl_' + str(lbl) + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + stem + '_lbl_' + str(lbl) + '_edges_2.vtp')

    print('\t\tPickling the graph as: ' + stem_pkl + '.pkl')
    ps.disperse_io.save_numpy(density, output_dir + '/' + stem + '_lbl_' + str(lbl) + '.vti')
    graph.pickle(output_dir + '/' + stem_pkl + '_lbl_' + str(lbl) + '.pkl')

    output_peaks = output_dir + '/' + stem + '_lbl_' + str(lbl) + '_peaks.vtp'
    print('\t\tStoring peaks cloud in file: ' + output_peaks)
    ps.disperse_io.save_vtp(tomo_peaks.to_vtp(), output_peaks)

    print('\t\tStoring peaks coordinates and rotations in file: ')
    write_mode = 'w'
    if cont > 0:
        write_mode = 'a'
    tomo_peaks.peaks_prop_op(ps.sub.PK_COORDS, gcrop_off, operator.add)
    if SWAP_XY:
        tomo_peaks.peaks_coords_swapxy()
        tomo_peaks.peaks_coords_swapxy(prop=OR_CLOSEST_N)
        tomo_peaks.vect_2pts(OR_CLOSEST_N, ps.sub.PK_COORDS, OR_NORMAL)
        tomo_peaks.vect_rotation_zrelion(key_v=OR_NORMAL, key_r=OR_ROT_A)
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_lbl_' + str(lbl) + '_edges_relion.vtp')
    tomo_peaks.save_coords(output_coords, swap_xy=False, add_prop=OR_ROT_A, fmode=write_mode)
    cont += 1

if len(unproc_l) == 0:
    print('All membranes were processed.')
    print('Successfully terminated. (' + time.strftime("%c") + ')')
elif len(unproc_l) == len(lbls):
    print('No membrane could be processed')
    print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')
else:
    print('WARNING: Membranes labled as ' + str(unproc_l) + ' could no be processed.')
    print('Successfully terminated. (' + time.strftime("%c") + ')')
