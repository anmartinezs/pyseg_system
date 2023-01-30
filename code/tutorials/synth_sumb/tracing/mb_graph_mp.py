"""

    Script for extracting an analyzing a GraphMCF from a segmented membrane (Multiprocessing version)

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
import argparse
import time
import sys
import pyseg as ps
import scipy as sp
import os
import numpy as np
import multiprocessing as mp

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
# INPUT PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '../../../..'  # Data path

# Input STAR file with segmentations
in_star = ROOT_PATH + '/data/tutorials/synth_sumb/segs/test_1_seg.star'
npr = 5 # number of parallel processes

####### Output data

output_dir = ROOT_PATH + '/data/tutorials/synth_sumb/graphs'

####### GraphMCF perameter

res = 1.048  # nm/pix
s_sig = 1.0  # 1.5
v_den = 0.0035  # 0.007 # 0.0025 # nm^3
ve_ratio = 4  # 2
max_len = 10  # 15 # 30 # nm

####### Advanced parameters

# nsig = 0.01
csig = 0.01
ang_rot = None
ang_tilt = None
nstd = 10  # 3
smooth = 3
mb_dst_off = 0.  # nm
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

# Get them from the command line if they were passed through itparser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--inStar', default=in_star, help='Input star file.')
parser.add_argument('--outDir', default=output_dir, help='Output files directory.')
parser.add_argument('--pixelSize', type=float, default=res, help='Resolution in nm/pix.')
parser.add_argument('--sSig', type=float, default=s_sig, help='Sigma for gaussian filtering.')
parser.add_argument('--vDen', type=float, default=v_den, help='Vertex density within membranes in cubic nm.')
parser.add_argument('--veRatio', type=float, default=ve_ratio,
                    help='Averaged ratio vertex/edge in the graph within membrane.')
parser.add_argument('--maxLen', type=float, default=max_len,
                    help='Maximum euclidean shortest distance to membrane in nm.')
parser.add_argument('-j', default=npr, type=int, help='Number of processors.')

args = parser.parse_args()
in_star = args.inStar
output_dir = args.outDir
res = args.pixelSize
s_sig = args.sSig
v_den = args.vDen
ve_ratio = args.veRatio
max_len = args.maxLen
npr = args.j

# Print initial message
print('Extracting GraphMCF and NetFilament objects from tomograms')
print('\tAuthor: ' + __author__)
print('\tDate: ' + time.strftime("%c") + '\n')
print('Options:')
# print '\tDisPerSe persistence threshold (nsig): ' + str(nsig)
print('\tSTAR file with the segmentations: ' + str(in_star))
print('\tNumber of parallel processes: ' + str(npr))
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
    print('\tTarget vertex density (membrane) ' +
          str(v_den) + ' vertex/nm^3 for property ' + v_prop + ' with mode ' + v_mode)
print('\tTarget edge/vertex ratio (non membrane) ' + str(ve_ratio) + ' for property ' + e_prop + ' with mode ' + e_mode)
if do_clahe:
    print('\t-Computing CLAHE.')
print('')

print('Paring input star file...')
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

        print('\tP[' + str(pr_id) + '] Sub-volume to process found: ' + input_tomo)
        print('\tP[' + str(pr_id) + '] Computing paths for ' + input_tomo + ' ...')
        path, stem_tomo = os.path.split(input_tomo)
        stem_pkl, _ = os.path.splitext(stem_tomo)
        input_file = output_dir + '/' + stem_pkl + '_g' + str(s_sig) + '.fits'
        _, stem = os.path.split(input_file)
        stem, _ = os.path.splitext(stem)

        print('\tP[' + str(pr_id) + '] Loading input data: ' + stem_tomo)
        tomo = ps.disperse_io.load_tomo(input_tomo).astype(np.float32)
        segh = ps.disperse_io.load_tomo(input_seg)

        print('\tP[' + str(pr_id) + '] Computing distance, mask and segmentation tomograms...')
        tomod = ps.disperse_io.seg_dist_trans(segh == SEG_MB) * res
        maskh = np.ones(shape=segh.shape, dtype=int)
        maskh[DILATE_NITER:-DILATE_NITER, DILATE_NITER:-DILATE_NITER, DILATE_NITER:-DILATE_NITER] = 0
        mask = np.asarray(tomod > (max_len + mb_dst_off + 2 * DILATE_NITER * res), dtype=int)
        maskh += mask
        maskh += (segh == 0)
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

        print('\tP[' + str(pr_id) + '] Smoothing input tomogram (s=' + str(s_sig) + ')...')
        density = sp.ndimage.filters.gaussian_filter(tomo, s_sig)
        density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1)
        ps.disperse_io.save_numpy(tomo, output_dir + '/' + stem_pkl + '.vti')
        ps.disperse_io.save_numpy(density.transpose(), input_file)
        ps.disperse_io.save_numpy(density, output_dir + '/' + stem + '.vti')

        print('\tP[' + str(pr_id) + '] Initializing DisPerSe...')
        work_dir = output_dir + '/disperse_pr_' + str(pr_id)
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
        rcut = round(density[mask_den].std() * csig, 4)
        print('\tP[' + str(pr_id) + '] Persistence cut thereshold set to: ' + str(rcut) + ' grey level')
        disperse.set_cut(rcut)
        disperse.set_mask(input_msk)
        disperse.set_smooth(smooth)

        print('\tP[' + str(pr_id) + '] Running DisPerSe...')
        disperse.mse(no_cut=False, inv=False)
        skel = disperse.get_skel()
        manifolds = disperse.get_manifolds(no_cut=False, inv=False)

        # Build the GraphMCF for the membrane
        print('\tP[' + str(pr_id) + '] Building MCF graph...')
        graph = ps.mb.MbGraphMCF(skel, manifolds, density, seg)
        graph.set_resolution(res)
        graph.build_from_skel(basic_props=False)
        graph.filter_self_edges()
        graph.filter_repeated_edges()

        print('\tP[' + str(pr_id) + '] Filtering nodes close to mask border...')
        mask = sp.ndimage.morphology.binary_dilation(mask, iterations=DILATE_NITER)
        for v in graph.get_vertices_list():
            x, y, z = graph.get_vertex_coords(v)
            if mask[int(round(x)), int(round(y)), int(round(z))]:
                graph.remove_vertex(v)
        print('\tP[' + str(pr_id) + '] Building geometry...')
        graph.build_vertex_geometry()

        if do_clahe:
            print('\tP[' + str(pr_id) + '] CLAHE on filed_value_inv property...')
            graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
            graph.clahe_field_value(max_geo_dist=50, N=256, clip_f=100., s_max=4.)

        print('\tP[' + str(pr_id) + '] Computing vertices and edges properties...')
        graph.compute_vertices_dst()
        graph.compute_edge_filamentness()
        graph.add_prop_inv(prop_topo, edg=True)
        graph.compute_edge_affinity()

        print('\tP[' + str(pr_id) + '] Applying general thresholds...')
        if ang_rot is not None:
            print('\tDeleting edges in MW area...')
            graph.filter_mw_edges(ang_rot, ang_tilt)

        print('\tP[' + str(pr_id) + '] Computing graph global statistics (before simplification)...')
        nvv, nev, nepv = graph.compute_global_stat(mask=mask_den)
        print('\t\t-P[' + str(pr_id) + '] Vertex density: ' + str(round(nvv, 5)) + ' nm^3')
        print('\t\t-P[' + str(pr_id) + '] Edge density: ' + str(round(nev, 5)) + ' nm^3')
        print('\t\t-P[' + str(pr_id) + '] Edge/Vertex ratio: ' + str(round(nepv, 5)))

        print('\tP[' + str(pr_id) + '] Graph density simplification for vertices...')
        if prop_topo != ps.globals.STR_FIELD_VALUE:
            print('\t\tProperty used: ' + prop_topo)
            graph.set_pair_prop(prop_topo)
        try:
            graph.graph_density_simp_ref(mask=np.asarray(mask_den, dtype=int), v_den=v_den,
                                         v_prop=v_prop, v_mode=v_mode)
        except ps.pexceptions.PySegInputWarning as e:
            print('P[' + str(pr_id) + '] WARNING: graph density simplification failed:')
            print('\t-' + e.get_message())

        print('\tP[' + str(pr_id) + '] Graph density simplification for edges in the membrane...')
        mask_mb = (seg == 1) * (~mask)
        nvv, nev, nepv = graph.compute_global_stat(mask=mask_mb)
        if nepv > ve_ratio:
            e_den = nvv * ve_ratio
            hold_e_prop = e_prop
            graph.graph_density_simp_ref(mask=np.asarray(mask_mb, dtype=int), e_den=e_den,
                                         e_prop=hold_e_prop, e_mode=e_mode, fit=True)

        print('\tP[' + str(pr_id) + '] Computing graph global statistics (after simplification)...')
        nvv, _, _ = graph.compute_global_stat(mask=mask_den)
        _, nev_mb, nepv_mb = graph.compute_global_stat(mask=mask_mb)
        print('\t\t-P[' + str(pr_id) + '] Vertex density (membrane): ' + str(round(nvv, 5)) + ' nm^3')
        print('\t\t-P[' + str(pr_id) + '] Edge density (membrane):' + str(round(nev_mb, 5)) + ' nm^3')
        print('\t\t-P[' + str(pr_id) + '] Edge/Vertex ratio (membrane): ' + str(round(nepv_mb, 5)))

        print('\tComputing graph properties (2)...')
        graph.compute_mb_geo(update=True)
        graph.compute_mb_eu_dst()
        graph.compute_edge_curvatures()
        graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
        graph.compute_vertices_dst()
        graph.compute_edge_filamentness()
        graph.compute_edge_affinity()

        print('\tSaving intermediate graphs...')
        ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                                output_dir + '/' + stem + '_edges.vtp')
        ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                                output_dir + '/' + stem + '_edges_2.vtp')
        # ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
        #                         output_dir + '/' + stem + '_sch.vtp')

        out_pkl = output_dir + '/' + stem_pkl + '.pkl'
        print('\tP[' + str(pr_id) + '] Pickling the graph as: ' + out_pkl)
        graph.pickle(out_pkl)
        # star.set_element('_psGhMCFPickle', row, out_pkl)
        q_pkls.put((row, out_pkl))
        pkls_dic[row] = out_pkl

    sys.exit(pr_id)

# Loop for processing the input data
print('Running main loop in parallel: ')
q_pkls = mp.Queue()
processes, pr_results = dict(), dict()
spl_ids = np.array_split(list(range(star.get_nrows())), npr)
for pr_id in range(npr):
    pr = mp.Process(target=pr_worker, args=(pr_id, spl_ids[pr_id], q_pkls))
    pr.start()
    processes[pr_id] = pr
for pr_id, pr in zip(iter(processes.keys()), iter(processes.values())):
    pr.join()
    if pr_id != pr.exitcode:
        print('ERROR: the process ' + str(pr_id) + ' ended unsuccessfully [' + str(pr.exitcode) + ']')
        print('Unsuccessfully terminated. (' + time.strftime("%c") + ')')

count, n_rows = 0, star.get_nrows()
while count < n_rows:
    hold_out_pkl = q_pkls.get()
    star.set_element(key='_psGhMCFPickle', row=hold_out_pkl[0], val=hold_out_pkl[1])
    count += 1

out_star = output_dir + '/' + os.path.splitext(os.path.split(in_star)[1])[0] + '_mb_graph.star'
print('\tStoring output STAR file in: ' + out_star)
star.store(out_star)

print('Terminated. (' + time.strftime("%c") + ')')
