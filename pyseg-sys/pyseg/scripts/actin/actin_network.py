"""

    Script for extracting the graphs for analyzing quantitatively an actin network

    Input:  - Density map tomogram
            - Segmentation tomogram of the synapse

    Output: - A graph

"""

__author__ = 'Antonio Martinez-Sanchez'

# ################ Package import

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

MSK_OFF = 0 # voxels

########################################################################################
# PARAMETERS
########################################################################################

####### Input data

ROOT_PATH = '/home/martinez/pool/pub/4Antonio/fromTillman' # '/home/martinez/pool/pool-lucic2/antonio/tomograms/marion/Clusters'

# Pre-processed density maps
in_tomo_l = (ROOT_PATH + '/in/t1_b2.fits',
             # ROOT_PATH + '/in/fits/pre/test_pre_cc.fits',
             # ROOT_PATH + '/in/fits/pre/test2_pre_cc.fits',
             # ROOT_PATH+'/in/fits/pre/20141124_W6t1_pre_cc.fits',
             # ROOT_PATH+'/in/fits/pre/20150930_W3t1_pre_cc.fits',
             # ROOT_PATH+'/in/fits/pre/20151116_W1t1_pre_cc.fits',
             # ROOT_PATH+'/in/fits/pre/20151116_W2t1_pre_cc.fits',
             # ROOT_PATH+'/in/fits/pre/20151116_W4t1_pre_cc.fits',
             # ROOT_PATH+'/in/fits/pre/20151116_W5t1_pre_cc.fits',
             # ROOT_PATH+'/in/fits/pre/20151216_W1t2_pre_cc.fits',
             # ROOT_PATH+'/in/fits/pre/20160125_FIB20151008_W2t1_pre_cc.fits',
	         # ROOT_PATH+'/in/fits/pre/20160125_FIB20151113_W2t1_pre_cc.fits',
	         # ROOT_PATH+'/in/fits/pre/20160125_FIB20160121_W2t1_pre_cc.fits',
             )

# Masks
in_mask_l = (ROOT_PATH + '/in/t1_b2_mask.fits',
             # ROOT_PATH + '/in/fits/pre/test_mask.fits',
             # ROOT_PATH + '/in/fits/pre/test_mask.fits',
             # ROOT_PATH+'/in/fits/pre/20141124_W6t1_mask.fits',
             # ROOT_PATH+'/in/fits/pre/20150930_W3t1_mask.fits',
             # ROOT_PATH+'/in/fits/pre/20151116_W1t1_mask.fits',
             # ROOT_PATH+'/in/fits/pre/20151116_W2t1_mask.fits',
             # ROOT_PATH+'/in/fits/pre/20151116_W4t1_mask.fits',
             # ROOT_PATH+'/in/fits/pre/20151116_W5t1_mask.fits',
             # ROOT_PATH+'/in/fits/pre/20151216_W1t2_mask.fits',
             # ROOT_PATH+'/in/fits/pre/20160125_FIB20151008_W2t1_mask.fits',
	         # ROOT_PATH+'/in/fits/pre/20160125_FIB20151113_W2t1_mask.fits',
	         # ROOT_PATH+'/in/fits/pre/20160125_FIB20160121_W2t1_mask.fits',
             )

# Threshold list
in_th_l = ((.0, 1.),
           # (.45, .47),
           # (.45, .47),
           # (.45, .47),
           # (.50, .55),
	       # (.5, .54),
	       # (.5, .54),
	       # (.55, .61),
	       # (.52, .55),
	       # (.60, .62),
	       # (.52, .56),
	       #  (.52, .56),
	       # (.52, .56),
           )

####### Output data

output_dir = ROOT_PATH+'/network/g'
out_stack_dir = ROOT_PATH+'/network/st'

####### GraphMCF

s_sig = 0.5
csig = 0.01
ang_rot = None # -96.9174
ang_tilt = None # 60
nstd = 3
smooth = 3
res = 2.736 # nm/pix

######## Masking thresholds
mask_off = 1 # nm
cc_neigh = 5 # nm

####### Graph density thresholds

v_den = 0.0004 # nm^3
ve_ratio = 2
v_prop = None # ps.globals.STR_FIELD_VALUE # In None topological simplification
e_prop = ps.globals.STR_FIELD_VALUE # ps.globals.STR_VERT_DST
v_mode = None # 'low'
e_mode = 'low'
prop_topo = None # None is ps.globals.STR_FIELD_VALUE

####### Graph global threholds

g_min_diam = 30 # nm
z_rang = [0, 90] # deg

####### Stack

verts = True # if True just verts are printed

########################################################################################
# MAIN ROUTINE
########################################################################################

# Print initial message
print 'Extracting GraphMCF from actin networks.'
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
print '\tOutput directory for stacks: ' + out_stack_dir
print 'Graph density thresholds:'
if v_prop is None:
    print '\tTarget vertex density (membranes) ' + str(v_den) + ' vertex/nm^3 for topological simplification'
else:
    print '\tTarget vertex density (membranes) ' + str(v_den) + ' vertex/nm^3 for property ' + v_prop + ' with mode ' + v_mode
print '\tTarget edge/vertex ratio (non membrane) ' + str(ve_ratio) + ' for property ' + e_prop + ' with mode ' + e_mode
print '\tNeighbor size for cross-correlation: ' + str(cc_neigh) + ' nm'
if verts:
    print '\tVertices will be stored in stacks.'
else:
    print '\tArcs will be stored in stacks.'
print ''

# Loop for processing the input data
print 'Running main loop: '
for (in_tomo, in_mask, in_th) in zip(in_tomo_l, in_mask_l, in_th_l):

    print '\tComputing paths for ' + in_tomo + ' ...'
    f_path, f_fname = os.path.split(in_tomo)
    f_stem_pkl, f_ext = os.path.splitext(f_fname)
    input_file = output_dir + '/' + f_stem_pkl + '_g' + str(s_sig) + '.fits'
    _, f_stem = os.path.split(input_file)
    f_stem, _ = os.path.splitext(f_stem)

    print '\tLoading input data: ' + f_fname
    tomo = ps.disperse_io.load_tomo(in_tomo).astype(np.float32)

    print '\tSmoothing input tomogram (s=' + str(s_sig) + ')...'
    density = scipy.ndimage.filters.gaussian_filter(tomo, s_sig)
    density = ps.globals.cont_en_std(density, nstd=nstd, lb=0, ub=1)

    print '\tStoring temporal tomograms for DisPerSe...'
    ps.disperse_io.save_numpy(tomo, output_dir + '/' + f_stem_pkl + '.vti')
    ps.disperse_io.save_numpy(density.transpose(), input_file)
    ps.disperse_io.save_numpy(density, output_dir + '/' + f_stem + '.vti')

    print '\tLoading mask: ' + in_mask
    mask = ps.disperse_io.load_tomo(in_mask).astype(np.bool)

    print '\tStoring temporal tomograms for VTK...'
    ps.disperse_io.save_numpy(density, output_dir + '/' + f_stem_pkl + '.vti')

    print '\tInitializing DisPerSe...'
    work_dir = output_dir + '/disperse'
    disperse = ps.disperse_io.DisPerSe(input_file, work_dir)
    try:
        disperse.clean_work_dir()
    except ps.pexceptions.PySegInputWarning as e:
        print e.get_message()

    # Manifolds for descending fields with the inverted image
    disperse.set_manifolds('J0a')
    # Down skeleton
    disperse.set_dump_arcs(-1)
    rcut = round(density[mask].std()*csig, 4)
    print '\tPersistence cut thereshold set to: ' + str(rcut) + ' grey level'
    disperse.set_cut(rcut)
    disperse.set_mask(in_mask)
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

    # print '\tFiltering vertices by field map (th=[' + str(in_th[0]) + ', ' + str(in_th[1]) + '])'
    # graph.threshold_vertices('cc', in_th[0], operator.lt)
    # graph.threshold_edges('cc', in_th[0], operator.lt)
    # graph.threshold_vertices('cc', in_th[1], operator.gt)
    # graph.threshold_edges('cc', in_th[1], operator.gt)

    print '\tFiltering nodes close to mask border...'
    mask = np.invert(mask)
    if MSK_OFF > 0:
        mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=MSK_OFF)
    for v in graph.get_vertices_list():
        x, y, z = graph.get_vertex_coords(v)
        try:
	    if not mask[int(round(x)), int(round(y)), int(round(z))]:
                graph.remove_vertex(v)
        except IndexError:
	    print 'WARNING: vertex out of bounds!'
            graph.remove_vertex(v)
    print '\t\tNumber of survivor vertices: ' + str(len(graph.get_vertices_list()))
    print '\tBuilding geometry...'
    graph.build_vertex_geometry()

    print '\tComputing vertices and edges properties...'
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()
    if prop_topo is None:
        graph.add_prop_inv(ps.globals.STR_FIELD_VALUE, edg=True)
    else:
        graph.add_prop_inv(prop_topo, edg=True)
    graph.compute_edge_affinity()

    print '\tApplying general thresholds...'
    if ang_rot is not None:
        print '\tDeleting edges in MW area...'
        graph.filter_mw_edges(ang_rot, ang_tilt)

    print '\tComputing graph global statistics (before simplification)...'
    nvv, nev, nepv = graph.compute_global_stat(mask=mask)
    print '\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3'
    print '\t\t-Edge density: ' + str(round(nev,5)) + ' nm^3'
    print '\t\t-Edge/Vertex ratio: ' + str(round(nepv,5))

    print '\tGraph density simplification for vertices...'
    if prop_topo != ps.globals.STR_FIELD_VALUE:
        print '\t\tProperty used: ' + ps.globals.STR_FIELD_VALUE
        graph.set_pair_prop(ps.globals.STR_FIELD_VALUE)
    try:
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
        graph.graph_density_simp_ref(mask=np.asarray(mask, dtype=np.int), e_den=e_den,
                                     e_prop=hold_e_prop, e_mode=e_mode, fit=True)
    else:
        print '\tWARNING: demanded ratio ' + str(ve_ratio) + ' could not be achieved (current is ' + str(nepv)

    print '\tComputing graph global statistics (after simplification)...'
    nvv, nev, pepv = graph.compute_global_stat(mask=mask)
    print '\t\t-Vertex density: ' + str(round(nvv,5)) + ' nm^3'
    print '\t\t-Edge density:' + str(round(nev,5)) + ' nm^3'
    print '\t\t-Edge/Vertex ratio: ' + str(round(nepv,5))

    print '\tCompute properties (2)...'
    graph.compute_edge_curvatures()
    graph.compute_edges_length(ps.globals.SGT_EDGE_LENGTH, 1, 1, 1, False)
    graph.compute_vertices_dst()
    graph.compute_edge_filamentness()
    graph.compute_edge_affinity()
    graph.compute_sgraph_relevance()
    graph.compute_diameters()
    graph.compute_edge_vectors()
    # graph.compute_vertex_vectors(30, key_dst=ps.disperse_io.STR_VERT_DST, fupdate=True)
    graph.compute_edge_zang()

    if z_rang is not None:
        print '\tFiltering edges with inclination in range:' + str(z_rang) +  ' deg'
        graph.threshold_edges(ps.disperse_io.STR_EDGE_ZANG, z_rang[0], operator.lt)
        graph.threshold_edges(ps.disperse_io.STR_EDGE_ZANG, z_rang[1], operator.gt)

    if g_min_diam is not None:
        print '\tThreshold by graph diameter (minimum): ' + str(g_min_diam) + ' nm'
        graph.threshold_vertices(ps.globals.STR_GRAPH_DIAM, g_min_diam, operator.lt)

    print '\tSaving intermediate graphs...'
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=True, edges=True),
                            output_dir + '/' + f_stem_pkl + '_edges.vtp')
    ps.disperse_io.save_vtp(graph.get_vtp(av_mode=False, edges=True),
                            output_dir + '/' + f_stem_pkl + '_edges_2.vtp')
    # ps.disperse_io.save_vtp(graph.get_scheme_vtp(nodes=True, edges=True),
    #                         output_dir + '/' + f_stem_pkl + '_sch.vtp')

    print '\tSaving the stack as: ' + f_stem_pkl + '_pts.mrc'
    ps.disperse_io.save_numpy(graph.to_mask(verts=verts).astype(np.int16),
                              out_stack_dir + '/' + f_stem_pkl + '_pts.mrc')

    print '\tPickling the graph as: ' + f_stem_pkl + '.pkl'
    graph.pickle(output_dir + '/' + f_stem_pkl + '.pkl')

print 'Terminated. (' + time.strftime("%c") + ')'