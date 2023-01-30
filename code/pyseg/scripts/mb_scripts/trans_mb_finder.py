"""

    Script for finding transmembrane high density structures (features)
    TODO: return feature oriented normal to membrane

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
import csv
import sys
import numpy as np
try:
    import pickle as pickle
except:
    import pickle

########## Global variables

SVOL_OFF = 3

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/fs/gpfs03/lv03/pool/pool-plitzko/Matthias/Tomography/161025_5mem_Rho'

# Original density map
input_tomo = ROOT_PATH + '/in/tomo16.mrc'

# Membrane segmentation: 1-mb, 2-cito, 3-ext (~=1-fg for non oriented normal analysis)
input_seg = ROOT_PATH + '/in/t16_5mem_lbl_gf1.5.mrc'

# List of key properties associated to features
prop_l = ('field_value_inv', 'ggf')

####### Output data

output_dir = ROOT_PATH + '/results'
out_log = output_dir + '/out.txt' # None then output is redirected to stdout

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
# AUXILIARY FUNCTIONALITY
########################################################################################

def store_list_feat_csv(graph, prop_l, crop_off, fname, mode):

    # Getting data
    prop_ids, d_types = list(), list()
    for prop in prop_l:
        prop_id = graph.get_prop_id(prop)
        n_comp = graph.get_prop_ncomp(key_id=prop_id)
        if n_comp != 1:
            print('ERROR: only properties with 1 component are valid, ' + prop + ' has ' + str(n_comp))
            sys.exit(-1)
        prop_ids.append(prop_id)
        d_types.append(ps.disperse_io.TypesConverter().gt_to_numpy(graph.get_prop_type(key_id=prop_id)))
    list_rows = list()
    list_cols = ('X', 'Y', 'Z') + prop_l
    for v in graph.get_vertices_list():
        data_l = list()
        data_l += list(np.asarray(graph.get_vertex_coords(v), dtype=float) + np.asarray(crop_off, dtype=float))
        for (prop_id, d_type)  in zip(prop_ids, d_types):
            data_l.append(graph.get_prop_entry_fast(prop_id, v.get_id(), 1, d_type)[0])
        list_rows.append(dict(list(zip(list_cols, data_l))))

    # Writing the .csv file
    if mode == 1:
        with open(fname, 'w') as ffile:
            writer = csv.DictWriter(ffile, fieldnames=list_cols)
            writer.writeheader()
            for row in list_rows:
                writer.writerow(row)
    else:
        with open(fname, 'a') as ffile:
            writer = csv.DictWriter(ffile, fieldnames=list_cols)
            for row in list_rows:
                writer.writerow(row)

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
# print '\tDisPerSe persistence threshold (nsig): ' + str(nsig)
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
output_csv = output_dir + '/' + stem_pkl + '_feat.csv'

print('\tLoading input data: ' + stem_tomo)
tomo = ps.disperse_io.load_tomo(input_tomo).astype(np.float32)
seg = ps.disperse_io.load_tomo(input_seg).astype(np.int16)
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
        print('\t\t\t-Local offset: ' + str(scrop_off))
        gcrop_off = tuple(np.asarray(crop_off) + np.asarray(scrop_off))
        print('\t\t\t-Global offset: ' + str(gcrop_off))

        print('\t\tComputing mask...')
        tomod = ps.disperse_io.seg_dist_trans(sseg == lbl) * res
        mask = np.asarray(tomod <= (mb_dst_off/res), dtype=bool)
        input_msk = output_dir + '/' + stem + '_lbl_' + str(lbl) + '_mask.fits'
        ps.disperse_io.save_numpy(np.asarray(~mask, dtype=float).transpose(), input_msk)

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

    print('\t\tSaving intermediate graphs...')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + stem + '_lbl_' + str(lbl) + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + stem + '_lbl_' + str(lbl) + '_edges_2.vtp')
    ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
                            output_dir + '/' + stem + '_lbl_' + str(lbl) + '_sch.vtp')

    print('\t\tPickling the graph as: ' + stem_pkl + '.pkl')
    ps.disperse_io.save_numpy(density, output_dir + '/' + stem + '_lbl_' + str(lbl) + '.vti')
    graph.pickle(output_dir + '/' + stem_pkl + '_lbl_' + str(lbl) + '.pkl')

    print('\t\tStoring the list of features in file ' + output_csv)
    if cont == 0:
        store_list_feat_csv(graph, prop_l, gcrop_off, output_csv, mode=1)
    else:
        store_list_feat_csv(graph, prop_l, gcrop_off, output_csv, mode=2)
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
